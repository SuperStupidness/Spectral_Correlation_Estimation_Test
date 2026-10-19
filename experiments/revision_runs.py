"""Long-running experiments for the paper revision.

Saves the per-trial values behind each metric (not just the averages) to
data/revision/, so trial-count convergence, variability and N/Np sensitivity
can all be worked out afterwards without re-running anything.

    python experiments/revision_runs.py            # SSCA / FAM ROC, leakage, validation
    python experiments/revision_runs.py --slow     # also the Fast-SC and 2D-FFT ROC runs
    python experiments/revision_runs.py --extra    # also other signal types, the window test
                                                   # and high-SNR leakage
    python experiments/revision_runs.py --list     # show the jobs and exit
    python experiments/revision_runs.py --smoke    # tiny end-to-end check

Finished jobs are skipped, so the script can be stopped and restarted.
The signals and detection statistics mirror scf_test/tests.py.
"""
import os

# One thread per worker process; the parallelism comes from running many jobs.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import scf_test.tests as test
from scf_test.algorithm import (ssca, fam, ssca_flattop, fam_flattop, fast_sc_wrapper,
                                scf_2d_fft_wrapper, scf_2d_fft_flattop)
from scf_test.psk import create_rect_bpsk_signal, create_srrc_qpsk_signal
from scf_test.fsk import create_gmsk_signal, create_cpfsk_signal

BASE_SEED = 2026
SNRS = (0, -5, -10)
SAMPLES_PER_SYMBOL = 10  # Symbol rate = 0.1 Hz
CFO = 0.05

# FAM hop size is Np/4, or Np/8 with the flattop window to avoid cycle aliasing
ESTIMATORS = {
    "ssca": (ssca, None),
    "ssca_flattop": (ssca_flattop, None),
    "fam": (fam, 4),
    "fam_flattop": (fam_flattop, 8),
    "fastsc": (fast_sc_wrapper, None),
    "2dfft": (scf_2d_fft_wrapper, None),
    "2dfft_flattop": (scf_2d_fft_flattop, None),
}
SSCA_FAM = ("ssca", "fam", "ssca_flattop", "fam_flattop")
SLOW = ("fastsc", "2dfft", "2dfft_flattop")  # seconds per call, where SSCA and FAM take milliseconds
NON_CONJUGATE_ONLY = ("fastsc", "fastsc_welch", "s3ca")

# Fast-SC with its coherence from the Welch PSD, like the other estimators, instead of its own option
VARIANTS = {"fastsc_welch": (partial(fast_sc_wrapper, builtin_coherence=False), None)}
SIGNALS = ("bpsk", "qpsk", "msk", "gmsk")

# As in tests.extended_window_test
WINDOWS = ["boxcar", "triang", ("kaiser", 4), "hamming", "hann", "cosine", "barthann", "lanczos", ("tukey", 0.2),
           ("chebwin", 100), ("kaiser", 14), "blackmanharris", "nuttall", "flattop", "parzen", "bohman", "taylor"]


def get_estimator(name, Np):
    if name == "s3ca":
        from algorithm.s3ca import s3ca
        return s3ca, {}
    func, hop_divisor = (ESTIMATORS | VARIANTS)[name]
    return func, ({} if hop_divisor is None else {"L": Np // hop_divisor})


def make_signal(kind, number_of_symbols, rng):
    """Noise-free test signal with the 0.05 Hz carrier offset."""
    if kind == "bpsk":
        return test.add_cfo(create_rect_bpsk_signal(number_of_symbols, SAMPLES_PER_SYMBOL, rng=rng), CFO)
    if kind == "qpsk":
        return test.add_cfo(create_srrc_qpsk_signal(number_of_symbols, SAMPLES_PER_SYMBOL, 21, 0.1, rng=rng), CFO)
    if kind == "msk":
        return create_cpfsk_signal(number_of_symbols, SAMPLES_PER_SYMBOL, modulation_index=0.5, f_carrier=CFO, rng=rng)[0]
    return create_gmsk_signal(number_of_symbols, SAMPLES_PER_SYMBOL, f_carrier=CFO, BT=0.3, gauss_filter_span=21, rng=rng)[0]


def leakage_region(alpha, N, conjugate):
    """Indices of the outputs between the strongest cycle frequency and its neighbour."""
    centre = 2*CFO if conjugate else 0
    mask = (np.abs(alpha - centre) > 1 / N) & (np.abs(alpha - centre) < 1/SAMPLES_PER_SYMBOL - 1 / N)
    return np.flatnonzero(mask)


def roc_job(job, rng):
    """Detection statistic of every trial, with and without the signal present.

    Same procedure as tests.plot_roc_non_conjugate / plot_roc_conjugate, but the
    statistics are kept so that the AUC, its spread and its convergence with the
    number of trials can be computed later.
    """
    func, kwargs = get_estimator(job["est"], job["Np"])
    N, snr, conjugate, trials = job["N"], job["snr"], job["conj"], job["trials"]

    if conjugate:
        names = ("bpsk", "msk", "gmsk")
        cycle_freqs = (2*CFO + 4/SAMPLES_PER_SYMBOL, CFO + 2/SAMPLES_PER_SYMBOL, CFO + 1/SAMPLES_PER_SYMBOL)
    else:
        names = ("bpsk", "qpsk", "gmsk")
        cycle_freqs = (4/SAMPLES_PER_SYMBOL, 1/SAMPLES_PER_SYMBOL, 1/SAMPLES_PER_SYMBOL)

    statistic = np.zeros((3, 2, trials))  # signal type, [present, absent], trial
    slices = None

    for j in range(2*trials):
        bpsk = test.add_cfo(create_rect_bpsk_signal(N, SAMPLES_PER_SYMBOL, rng=rng), CFO)
        if conjugate:
            second, _, _ = create_cpfsk_signal(N, SAMPLES_PER_SYMBOL, modulation_index=0.5, f_carrier=CFO, rng=rng)
        else:
            second = test.add_cfo(create_srrc_qpsk_signal(N, SAMPLES_PER_SYMBOL, 21, 0.1, rng=rng), CFO)
        gmsk, _, _ = create_gmsk_signal(N, SAMPLES_PER_SYMBOL, f_carrier=CFO, BT=0.3, gauss_filter_span=21, rng=rng)

        # Even for signal present, odd for noise only
        present = j % 2 == 0

        for s, clean in enumerate((bpsk, second, gmsk)):
            noisy = test.add_awgn_snr(clean, desired_snr=snr, rng=rng)
            signal = noisy[:N] if present else (noisy - clean)[:N]

            coherence, _, alpha = func(signal, Np=job["Np"], conjugate=conjugate, coherence=True, **kwargs)

            # The alpha grid is the same on every call, so locate the slices once
            if slices is None:
                resolution = test.get_alpha_resolution(alpha) / 2
                slices = [np.flatnonzero(np.abs(alpha - cf) <= resolution) for cf in cycle_freqs]

            statistic[s, 0 if present else 1, j // 2] = np.max(np.abs(coherence.ravel()[slices[s]]))

    return {"statistic": statistic, "signals": np.array(names), "cycle_freqs": np.array(cycle_freqs)}


def leakage_job(job, rng):
    """Mean SCD magnitude over the cycle leakage region for every run (tests.window_test)."""
    func, kwargs = get_estimator(job["est"], job["Np"])
    N, conjugate, trials = job["N"], job["conj"], job["trials"]
    sparse = job["est"] == "s3ca"  # output points change from run to run

    per_run = np.zeros(trials)
    region = None

    for j in range(trials):
        bpsk = test.add_cfo(create_rect_bpsk_signal(int(N / 8), SAMPLES_PER_SYMBOL, rng=rng), CFO)
        signal = test.add_awgn_snr(bpsk, desired_snr=job["snr"], rng=rng)[:N]

        scd, _, alpha = func(signal, Np=job["Np"], conjugate=conjugate, coherence=False, **kwargs)
        scd = np.abs(scd)

        if region is None or sparse:
            region = leakage_region(alpha, N, conjugate)

        per_run[j] = np.mean(scd.ravel()[region]) if len(region) > 0 else np.nan

    return {"per_run": per_run}


def windows_job(job, rng):
    """Cycle leakage of every window choice on one signal type (tests.extended_window_test).

    SSCA and FAM take a pair of windows, a(n) and g(n); the 2D-FFT takes one lag window.
    """
    N, Np, conjugate, trials = job["N"], job["Np"], job["conj"], job["trials"]
    single = job["est"] == "2dfft"
    shape = (len(WINDOWS),) if single else (len(WINDOWS), len(WINDOWS))

    per_run = np.zeros(shape + (trials,))
    region = None

    for j in range(trials):
        signal = test.add_awgn_snr(make_signal(job["signal"], int(N / 8), rng), desired_snr=job["snr"], rng=rng)[:N]

        for index in np.ndindex(shape):
            if single:
                scd, _, alpha = scf_2d_fft_wrapper(signal, Np=Np, conjugate=conjugate, window=WINDOWS[index[0]])
            elif job["est"] == "ssca":
                scd, _, alpha = ssca_flattop(signal, Np=Np, conjugate=conjugate,
                                             window_1=WINDOWS[index[0]], window_2=WINDOWS[index[1]])
            else:
                scd, _, alpha = fam_flattop(signal, Np=Np, L=Np // 4, conjugate=conjugate,
                                            window_1=WINDOWS[index[0]], window_2=WINDOWS[index[1]])

            if region is None:
                region = leakage_region(alpha, N, conjugate)

            per_run[index + (j,)] = np.mean(np.abs(scd).ravel()[region])

    return {"per_run": per_run, "windows": np.array([str(window) for window in WINDOWS])}


def cdp_job(job, rng):
    """Cyclic domain profile, max over f of |coherence|, of every realization of one signal type."""
    func, kwargs = get_estimator(job["est"], job["Np"])
    N = job["N"]
    result = {}

    for j in range(job["trials"]):
        signal = test.add_awgn_snr(make_signal(job["signal"], int(N / 8), rng), desired_snr=job["snr"], rng=rng)[:N]
        coherence, _, alpha = func(signal, Np=job["Np"], conjugate=job["conj"], coherence=True, **kwargs)
        # The sparse S3CA output has different cycle frequencies in every realization
        result[f"alpha_{j}"], result[f"profile_{j}"] = test.compute_cdp(coherence, alpha)

    return result


def validation_job(job, rng, out_dir):
    """RMSE against the theoretical BPSK SCD for each trial and signal length.

    Both the magnitudes and the complex values are compared. One trial takes
    minutes at the longest signal lengths, so every trial is written out as it
    finishes and picked up again on a restart.
    """
    func, kwargs = get_estimator(job["est"], job["Np"])
    parts = [out_dir / job["kind"] / f"{job['name']}_trial{k:02d}.npy" for k in range(job["trials"])]
    restarted = any(part.exists() for part in parts)
    rmse = []

    for k, part in enumerate(parts):
        if not part.exists():
            # A restarted job starts its random stream afresh, which would repeat the signals of
            # the trials it already has; its remaining trials get a seed of their own instead
            trial_rng = np.random.default_rng([BASE_SEED, zlib.crc32(job["name"].encode()), k]) if restarted else rng
            magnitude, _, complex_values = test.validation_test(func, Np=job["Np"], max_log_2=job["max_log_2"], no_of_run=1,
                                                                alpha_max=job.get("alpha_max", 1.0),
                                                                extra_samples=job.get("extra_samples", 0),
                                                                conjugate=job["conj"], plot=False, save=False,
                                                                fam="L" in kwargs, rng=trial_rng, complex_rmse=True)
            np.save(part, np.array([magnitude, complex_values]))
        rmse.append(np.load(part))

    rmse = np.array(rmse)  # trial, [magnitude, complex], signal length
    return {"rmse": rmse[:, 0], "rmse_complex": rmse[:, 1], "signal_length": 2**np.arange(10, job["max_log_2"])}


def run_job(job, out_dir, code_version):
    start = time.perf_counter()
    path = out_dir / job["kind"] / f"{job['name']}.npz"
    if path.exists():
        return "skipped", 0.0

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Seed depends only on the job, so results do not depend on scheduling
        # A job can borrow another job's seed, to see exactly the same signals
        rng = np.random.default_rng([BASE_SEED, zlib.crc32(job.get("seed_as", job["name"]).encode())])

        if job["kind"] == "roc":
            result = roc_job(job, rng)
        elif job["kind"] == "leakage":
            result = leakage_job(job, rng)
        elif job["kind"] == "windows":
            result = windows_job(job, rng)
        elif job["kind"] == "cdp":
            result = cdp_job(job, rng)
        else:
            result = validation_job(job, rng, out_dir)

        elapsed = time.perf_counter() - start
        settings = {key: np.array(value) for key, value in job.items()}
        partial = path.with_suffix(".partial")
        with open(partial, "wb") as file:
            np.savez(file, **result, **settings, code_version=np.array(code_version),
                     base_seed=np.array(BASE_SEED), elapsed_seconds=np.array(elapsed))
        for attempt in range(10):  # a sync client can hold the new file open for a moment
            try:
                os.replace(partial, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(2)
        return "done", elapsed
    except Exception as error:
        return f"FAILED: {error!r}", time.perf_counter() - start


def build_jobs(smoke=False, slow=False, extra=False):
    jobs = []

    def add(kind, est, Np, N, conj, trials, snr=0, tag=None, **settings):
        if conj and est in NON_CONJUGATE_ONLY:
            return
        name = f"{kind}_{est}_Np{Np}_N{N}_snr{snr}_{'c' if conj else 'nc'}" + (f"_{tag}" if tag else "")
        jobs.append(dict(kind=kind, name=name, est=est, Np=Np, N=N, snr=snr, conj=conj, trials=trials, **settings))

    if smoke:
        for conj in (False, True):
            for est in ESTIMATORS:
                add("roc", est, 16, 1024, conj, 3)
                add("leakage", est, 16, 1024, conj, 3)
                add("cdp", est, 16, 1024, conj, 2, snr=10, tag="gmsk", signal="gmsk")
            add("roc", "fastsc_welch", 16, 1024, conj, 3, seed_as="roc_fastsc_Np16_N1024_snr0_nc")
            add("validation", "ssca", 128, 2048, conj, 2, snr=10, tag="complex", max_log_2=12)
            add("validation", "fam", 128, 2048, conj, 2, snr=10, tag="complex", max_log_2=12)
            for est in ("ssca", "fam", "2dfft"):
                add("windows", est, 16, 1024, conj, 1, tag="bpsk", signal="bpsk")
        for N in (4096, 16384, 65536):
            add("leakage", "s3ca", 64, N, False, 3)
        for signal in SIGNALS:
            add("cdp", "s3ca", 64, 4096, False, 2, snr=10, tag=signal, signal=signal)
            add("windows", "ssca", 16, 1024, False, 1, tag=signal, signal=signal)
        add("validation", "fastsc", 64, 2048, False, 1, snr=10, tag="complex", max_log_2=12, alpha_max=0.5)
        return jobs

    # Cycle leakage: cheap, covers every estimator over N and Np
    for conj in (False, True):
        for N in (1024, 4096, 16384, 65536):
            for Np in (16, 64, 256):
                for est in SSCA_FAM:
                    add("leakage", est, Np, N, conj, 100)
                if Np**2 * N <= 64**2 * 65536:  # Fast-SC run time grows with Np^2 N
                    add("leakage", "fastsc", Np, N, conj, 100)
                if N <= 4096:  # 2D-FFT needs O(N^2) memory
                    add("leakage", "2dfft", Np, N, conj, 100)
                    add("leakage", "2dfft_flattop", Np, N, conj, 100)
            if N >= 4096:  # the S3CA implementation fails on shorter signals
                add("leakage", "s3ca", 64, N, conj, 100)

    # ROC at the paper's setting (N = 4096, Np = 64), then its sensitivity to N and Np
    for conj in (False, True):
        for snr in SNRS:
            for est in SSCA_FAM:
                add("roc", est, 64, 4096, conj, 500, snr=snr)
    for conj in (False, True):
        for snr in SNRS:
            for est in SSCA_FAM:
                for N in (1024, 16384, 65536):
                    add("roc", est, 64, N, conj, 500, snr=snr)
                for Np in (16, 256):
                    add("roc", est, Np, 4096, conj, 500, snr=snr)

    # The same for the slow estimators, hours per job at the longer signal lengths
    if slow:
        for conj in (False, True):
            for snr in SNRS:
                for est in SLOW:
                    add("roc", est, 64, 4096, conj, 500, snr=snr)
                # Fast-SC with the Welch normalization, on the same signals as the Fast-SC job above
                add("roc", "fastsc_welch", 64, 4096, conj, 500, snr=snr, seed_as=f"roc_fastsc_Np64_N4096_snr{snr}_nc")
        for conj in (False, True):
            for snr in SNRS:
                for N in (1024, 16384):
                    add("roc", "fastsc", 64, N, conj, 500, snr=snr)
                for N in (1024, 2048):
                    add("roc", "2dfft", 64, N, conj, 500, snr=snr)
                    add("roc", "2dfft_flattop", 64, N, conj, 500, snr=snr)
        # Fast-SC over Np for the Pareto front (non-conjugate only); longest first, as Np = 256 takes many hours
        for Np in (256, 128, 32, 16, 8):
            for snr in SNRS:
                add("roc", "fastsc", Np, 4096, False, 500, snr=snr)

    # Validation against theory, signal lengths 2^10 .. 2^19
    for conj in (False, True):
        for est in ("ssca", "fam"):
            add("validation", est, 512, 2**19, conj, 20, snr=10, tag="complex", max_log_2=20)
    # Fast-SC takes hours per trial at Np = 512 and covers alpha up to 1/2 only. Its wrapper
    # zero-pads the signal so that its cycle frequency resolution is 1/N like the others
    # Twenty trials like the others, in four batches of five that can run side by side (--match)
    for tag in ("complex", "complex_b2", "complex_b3", "complex_b4"):
        add("validation", "fastsc", 512, 2**19, False, 5, snr=10, tag=tag, max_log_2=20, alpha_max=0.5)

    if extra:
        # Cycle leakage at high SNR, where little but the leakage itself is left
        for conj in (False, True):
            for est in (*ESTIMATORS, "s3ca"):
                add("leakage", est, 64, 4096, conj, 100, snr=30)

        # Cyclic domain profile of each signal type, at the setting of the paper's coherence test
        for signal in SIGNALS:
            for conj in (False, True):
                for est in (*ESTIMATORS, "s3ca"):
                    add("cdp", est, 64, 4096, conj, 20, snr=10, tag=signal, signal=signal)

        # Window test on each signal type. Only BPSK has a conjugate feature at 2 f_CFO to leak from.
        for signal in SIGNALS:
            for conj in ((False, True) if signal == "bpsk" else (False,)):
                for est in ("ssca", "fam", "2dfft"):
                    add("windows", est, 64, 4096, conj, 100, tag=signal, signal=signal)
                # Longer signal, to see whether the best windows depend on N (2D-FFT needs O(N^2) memory).
                # Fewer runs: each takes eight times longer, and every window pair sees the same signals.
                for est in ("ssca", "fam"):
                    add("windows", est, 64, 32768, conj, 25, tag=signal, signal=signal)

    return jobs


def get_code_version():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "revision", help="output directory")
    parser.add_argument("--workers", type=int, default=8, help="number of parallel ROC / leakage jobs")
    parser.add_argument("--only", choices=("roc", "leakage", "validation", "windows", "cdp"),
                        action="append", help="run one kind of job")
    parser.add_argument("--list", action="store_true", help="show the jobs and exit")
    parser.add_argument("--smoke", action="store_true", help="tiny version of every kind of job")
    parser.add_argument("--slow", action="store_true", help="include the Fast-SC and 2D-FFT ROC runs")
    parser.add_argument("--extra", action="store_true",
                        help="include other signal types, the window test and high-SNR leakage")
    parser.add_argument("--log", default="log.txt", help="log file name inside the output directory")
    parser.add_argument("--match", default="", help="only run jobs whose name contains this text")
    parser.add_argument("--name", help="only run the job with exactly this name")
    args = parser.parse_args()

    jobs = [job for job in build_jobs(args.smoke, args.slow, args.extra)
            if (not args.only or job["kind"] in args.only) and args.match in job["name"]
            and (not args.name or job["name"] == args.name)]
    if args.list:
        for job in jobs:
            print(job["name"])
        print(f"{len(jobs)} jobs")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    code_version = get_code_version()
    log_path = args.out / args.log

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")

    log(f"{len(jobs)} jobs, {args.workers} workers, code version {code_version}, output in {args.out}")

    # Validation needs tens of GB at the longest signal lengths, so it runs one
    # job at a time next to the others. S3CA is native code that can take its
    # process down with it, so every S3CA job gets a process of its own.
    pools = {"light": ProcessPoolExecutor(args.workers), "heavy": ProcessPoolExecutor(1)}
    futures = {}
    for job in jobs:
        pool = "heavy" if job["kind"] == "validation" else job["name"] if job["est"] == "s3ca" else "light"
        if pool not in pools:
            pools[pool] = ProcessPoolExecutor(1)
        futures[pools[pool].submit(run_job, job, args.out, code_version)] = job

    finished = 0
    for future in as_completed(futures):
        finished += 1
        try:
            status, elapsed = future.result()
        except Exception as error:  # the worker process itself died
            status, elapsed = f"CRASHED: {error!r}", 0.0
        log(f"[{finished}/{len(jobs)}] {futures[future]['name']}: {status} ({elapsed:.0f} s)")

    for pool in pools.values():
        pool.shutdown()
    log("finished")


if __name__ == "__main__":
    main()
