"""Tables from the per-trial results of revision_runs.py.

    python experiments/revision_summary.py

Writes CSV files to data/revision/summary/ and prints the rows at the paper's
setting (N = 4096, Np = 64). Works on whatever jobs have finished so far.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPS = 1000
OUTLIER = 100  # same threshold as tests.window_test


def load(folder):
    jobs = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as data:
            jobs.append({key: (data[key].item() if data[key].ndim == 0 else data[key]) for key in data.files})
    return jobs


def auc(present, absent):
    """Area under the ROC: chance that a signal-present statistic beats a noise-only one."""
    ranks = rankdata(np.concatenate((present, absent)))
    n_p, n_a = len(present), len(absent)
    return (ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_a)


def mean_auc_with_interval(statistic, rng):
    """Mean AUC over the three signal types, with a 95% bootstrap interval over trials."""
    trials = statistic.shape[2]
    per_signal = [auc(s[0], s[1]) for s in statistic]
    resampled = np.zeros(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        picks = rng.integers(0, trials, (len(statistic), 2, trials))
        resampled[b] = np.mean([auc(s[0][p[0]], s[1][p[1]]) for s, p in zip(statistic, picks)])
    low, high = np.percentile(resampled, [2.5, 97.5])
    return per_signal, float(np.mean(per_signal)), low, high


def settings(job):
    return {"estimator": job["est"], "Np": job["Np"], "N": job["N"], "snr_db": job["snr"],
            "scd": "conjugate" if job["conj"] else "non-conjugate", "code_version": job["code_version"]}


def roc_tables(jobs, rng):
    rows, convergence = [], []
    for job in jobs:
        per_signal, mean, low, high = mean_auc_with_interval(job["statistic"], rng)
        row = settings(job) | {"trials": job["trials"], "mean_auc": mean, "ci95_low": low, "ci95_high": high}
        row |= {f"auc_{name}": value for name, value in zip(job["signals"], per_signal)}
        rows.append(row)

        # How the result settles as trials are added (paper setting only)
        if job["N"] == 4096 and job["Np"] == 64:
            for n in (25, 50, 100, 200, 300, 400, 500):
                if n <= job["trials"]:
                    _, mean_n, low_n, high_n = mean_auc_with_interval(job["statistic"][:, :, :n], rng)
                    convergence.append(settings(job) | {"trials": n, "mean_auc": mean_n,
                                                        "ci95_low": low_n, "ci95_high": high_n})
    return pd.DataFrame(rows), pd.DataFrame(convergence)


def leakage_table(jobs):
    rows = []
    for job in jobs:
        runs = job["per_run"]
        valid = runs[~np.isnan(runs)]
        q1, median, q3 = np.percentile(valid, [25, 50, 75])
        rows.append(settings(job) | {"runs": len(runs), "mean": valid.mean(), "std": valid.std(ddof=1),
                                     "median": median, "q1": q1, "q3": q3, "max": valid.max(),
                                     "runs_over_100": int(np.sum(valid > OUTLIER)), "runs_nan": int(np.isnan(runs).sum()),
                                     "mean_first_10": valid[:10].mean(), "mean_first_25": valid[:25].mean(),
                                     "mean_first_50": valid[:50].mean()})
    return pd.DataFrame(rows)


def pool_batches(jobs):
    """Join the validation jobs that were run as several batches (name, name_b2, name_b3, ...)."""
    pooled = {}
    for job in jobs:
        base = re.sub(r"_b\d+$", "", job["name"])
        if base not in pooled:
            pooled[base] = dict(job) | {"name": base}
            continue
        first = pooled[base]
        for key in ("rmse", "rmse_complex"):
            if key in first:
                first[key] = np.concatenate((first[key], job[key]))
        first["trials"] = len(first["rmse"])
    return list(pooled.values())


def validation_table(jobs):
    rows = []
    for job in pool_batches(jobs):
        rmse = job["rmse"]  # trial, signal length
        for i, length in enumerate(job["signal_length"]):
            row = settings(job) | {"alpha_max": job.get("alpha_max", 1.0),
                                   "signal_length": int(length), "trials": len(rmse),
                                   "rmse_mean": rmse[:, i].mean(), "rmse_std": rmse[:, i].std(ddof=1)}
            # RMSE pooled over the first n trials, as validation_test does with no_of_run=n
            row |= {f"rmse_pooled_{n}": np.sqrt(np.mean(rmse[:n, i]**2)) for n in (1, 5, 10, 20) if n <= len(rmse)}
            if "rmse_complex" in job:  # complex values compared, not only magnitudes
                row |= {"rmse_complex_mean": job["rmse_complex"][:, i].mean(),
                        "rmse_complex_std": job["rmse_complex"][:, i].std(ddof=1)}
            rows.append(row)
    return pd.DataFrame(rows)


def windows_table(jobs):
    """Best window choice per signal type, and where the paper's choice (flattop, with boxcar g(n)) ranks."""
    rows = []
    for job in jobs:
        mean = job["per_run"].mean(axis=-1)
        names = [str(name) for name in job["windows"]]
        best = np.unravel_index(np.argmin(mean), mean.shape)
        if mean.ndim == 2:  # SSCA and FAM: a(n), g(n)
            paper, hamming = (names.index("flattop"), names.index("boxcar")), (names.index("hamming"),) * 2
        else:  # 2D-FFT: one lag window
            paper, hamming = (names.index("flattop"),), (names.index("hamming"),)
        rows.append(settings(job) | {"signal": job["signal"], "runs": job["trials"],
                                     "best": " / ".join(names[i] for i in best), "best_leakage": mean[best],
                                     "flattop_leakage": mean[paper], "flattop_rank": int(np.sum(mean < mean[paper])) + 1,
                                     "hamming_leakage": mean[hamming], "choices": mean.size})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "revision", help="output directory of revision_runs.py")
    args = parser.parse_args()

    out = args.data / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)

    roc, convergence = roc_tables(load(args.data / "roc"), rng)
    leakage = leakage_table(load(args.data / "leakage"))
    validation = validation_table(load(args.data / "validation"))
    fastsc_validation = validation_table(load(args.data / "fastsc_validation"))
    windows = windows_table(load(args.data / "windows"))

    for name, table in (("roc_auc", roc), ("roc_auc_convergence", convergence),
                        ("leakage", leakage), ("validation_rmse", validation),
                        ("fastsc_validation_rmse", fastsc_validation), ("windows", windows)):
        if not table.empty:
            table.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")

    def paper_setting(table):
        return table[(table["N"] == 4096) & (table["Np"] == 64)].drop(columns=["N", "Np", "code_version"])

    if not roc.empty:
        print("\nROC, N = 4096, Np = 64")
        print(paper_setting(roc).sort_values(["scd", "snr_db", "estimator"], ascending=[False, False, True]).to_string(index=False))
    if not leakage.empty:
        print("\nCycle leakage, N = 4096, Np = 64")
        columns = ["estimator", "scd", "snr_db", "mean", "std", "median", "q1", "q3", "max", "runs_over_100"]
        print(paper_setting(leakage).sort_values(["snr_db", "scd", "estimator"], ascending=[True, False, True])[columns].to_string(index=False))
    if not windows.empty:
        print("\nWindow test: best choice per signal type, and the rank of flattop (with boxcar g(n))")
        print(windows.drop(columns=["N", "Np", "code_version"]).sort_values(["estimator", "scd", "signal"]).to_string(index=False))
    if not validation.empty:
        print("\nValidation RMSE against theory")
        print(validation.drop(columns=["N", "snr_db", "code_version", "alpha_max"]).to_string(index=False))
    if not fastsc_validation.empty:
        print("\nFast-SC validation RMSE against theory (Np = 128; cycle frequencies below alpha_max)")
        print(fastsc_validation.drop(columns=["N", "snr_db", "code_version"]).to_string(index=False))


if __name__ == "__main__":
    main()
