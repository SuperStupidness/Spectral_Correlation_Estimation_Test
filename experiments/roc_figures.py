"""Figs. 6 and 7 of the paper (robustness test and Pareto front), redrawn with the corrected Fast-SC wrapper.

    python experiments/revision_summary.py     # writes data/revision/summary/roc_auc.csv
    python experiments/roc_figures.py

Same layout as the cells of roc_test_extra.ipynb that drew the originals.

Fig. 6 takes every curve from the revision runs (N = 4096, Np = 64, 500 trials), and so does Fig. 7
for every configuration and Np (the notebook's scores used the GMSK generator before its fix). Execution
times are the minima from experiments/pareto_benchmark.py, every configuration measured in one session
with the current code (the notebook's older timings had unreliable minima for the FAM at Np >= 64).

Writes fig/roc_compare_subplot_rev_2.png and fig/pareto_front_rev.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textalloc as ta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_summary import auc

ROOT = Path(__file__).resolve().parents[1]
SNR = [0, -5, -10]
AXIS_GREY = "#898781"
NPS = [8, 16, 32, 64, 128, 256]
# Drawn at the printed size (as included in the paper: Fig. 6 9 x 5 cm, Fig. 7 9 x 7 cm), so the fonts
# print at the sizes set here and LaTeX does not rescale or stretch the figures
CM = 1 / 2.54
FIG6_SIZE, FIG7_SIZE = (9 * CM, 5 * CM), (9 * CM, 7 * CM)
TICK, LABEL, SMALL = 6.5, 7, 6


def roc_figure():
    table = pd.read_csv(ROOT / "data" / "revision" / "summary" / "roc_auc.csv")
    table = table[(table["N"] == 4096) & (table["Np"] == 64)]

    def scores(estimator, scd):
        rows = table[(table["estimator"] == estimator) & (table["scd"] == scd)].set_index("snr_db")
        return [rows.loc[snr, "mean_auc"] for snr in SNR]

    lines = [("fam", "FAM", "red", "o"), ("ssca", "SSCA", "blue", "x"), ("fam_flattop", "FAM flattop", "orange", "s"),
             ("ssca_flattop", "SSCA flattop", "green", "^"), ("fastsc", "FastSC", "brown", "."),
             ("2dfft", "2D-FFT", "pink", "*"), ("2dfft_flattop", "2D-FFT flattop", "purple", "+")]

    fig, axes = plt.subplots(1, 2, figsize=FIG6_SIZE)
    for ax, scd in zip(axes, ("non-conjugate", "conjugate")):
        for estimator, label, color, marker in lines:
            if scd == "conjugate" and estimator == "fastsc":
                continue
            ax.plot(SNR, scores(estimator, scd), label=label, marker=marker, linestyle="--", color=color,
                    linewidth=0.9, markersize=3.5)
        ax.set_xlim(0.2, -10.2)
        ax.set_xticks(SNR)
        ax.set_xlabel("SNR (dB)", fontsize=LABEL, labelpad=1)
        ax.tick_params(labelsize=TICK, length=2, pad=1.5)
    axes[0].set_ylabel("Total AUC Score (Normalized)", fontsize=SMALL, labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=SMALL, bbox_to_anchor=(0.5, 1.0), frameon=False,
               columnspacing=0.9, handletextpad=0.3, handlelength=1.8, borderaxespad=0.1)
    fig.subplots_adjust(left=0.11, right=0.955, bottom=0.17, top=0.79, wspace=0.24)
    return fig


def pareto_score(estimator, Np):
    """Non-conjugate mean AUC over the three signals and the three SNRs, from the revision runs."""
    values = []
    for snr in SNR:
        path = ROOT / "data" / "revision" / "roc" / f"roc_{estimator}_Np{Np}_N4096_snr{snr}_nc.npz"
        statistic = np.load(path)["statistic"]  # missing until revision_runs.py has finished them
        second = path.with_name(path.stem + "_b2.npz")  # second batch of trials (--second-batch), pooled
        if second.exists():
            statistic = np.concatenate((statistic, np.load(second)["statistic"]), axis=2)
        values.append(np.mean([auc(present, absent) for present, absent in statistic]))
    return np.mean(values)


def find_pareto_front(auc_values, times, tolerance=0.05):
    """Points that no other point beats in both AUC (higher) and time (lower).

    Times within `tolerance` of each other count as equal: the 2D-FFT takes the same time at every Np,
    and differences of a few percent are measurement noise.
    """
    auc_values, times = np.asarray(auc_values), np.asarray(times)
    no_slower = times[None, :] <= times[:, None] * (1 + tolerance)  # j is not meaningfully slower than i
    faster = times[None, :] * (1 + tolerance) < times[:, None]       # j is meaningfully faster than i
    dominated = ((auc_values[:, None] <= auc_values[None, :]) & no_slower
                 & ((auc_values[:, None] < auc_values[None, :]) | faster))
    np.fill_diagonal(dominated, False)
    return ~dominated.any(axis=1)


def benchmark_minima():
    """Minimum execution time per (configuration, Np), from experiments/pareto_benchmark.py."""
    with open(ROOT / "data" / "revision" / "pareto_benchmark.json") as file:
        runs = json.load(file)["benchmarks"]
    return {(run["extra_info"]["configuration"], run["extra_info"]["Np"]): run["stats"]["min"] for run in runs}


def pareto_figure():
    minima = benchmark_minima()
    # Configuration (revision runs and benchmark), label
    algorithms = [("ssca", "SSCA_{}"), ("fam", "FAM_{}"), ("ssca_flattop", "SSCA_{}f"), ("fam_flattop", "FAM_{}f"),
                  ("fastsc", "FastSC_{}"), ("2dfft", "2D-FFT_{}"), ("2dfft_flattop", "2D-FFT_{}f")]
    scores, times, names = [], [], []
    for name, label in algorithms:
        for Np in NPS:
            scores.append(pareto_score(name, Np))
            times.append(minima[(name, Np)])
            names.append(label.format(Np))
    scores, times = np.array(scores), np.array(times)
    front = find_pareto_front(scores, times)

    # Colour and marker per configuration as in Fig. 6 (roc_figure); each configuration's points are
    # joined in order of Np and labelled with Np only, which keeps the labels short enough not to collide.
    # Same order as `algorithms` above.
    families = [("SSCA", "blue", "x"), ("FAM", "red", "o"), ("SSCA flattop", "green", "^"), ("FAM flattop", "orange", "s"),
                ("FastSC", "brown", "."), ("2D-FFT", "pink", "*"), ("2D-FFT flattop", "purple", "+")]
    fig, ax = plt.subplots(figsize=FIG7_SIZE)
    ax.set_xscale("log", base=2)
    order = np.argsort(times[front])
    ax.plot(times[front][order], scores[front][order], color="black", linestyle="--", linewidth=0.8, alpha=0.8,
            zorder=1, label="Pareto front")
    ax.scatter(times[front], scores[front], s=45, facecolors="none", edgecolors="black", linewidths=0.7, zorder=4)
    for i, (label, color, marker) in enumerate(families):
        part = slice(i * len(NPS), (i + 1) * len(NPS))
        ax.plot(times[part], scores[part], color=color, linewidth=0.6, alpha=0.5, zorder=2)
        ax.plot(times[part], scores[part], linestyle="none", marker=marker, ms=3.5, color=color, mew=0.9, label=label,
                zorder=3)

    labels = np.array([str(Np) for _ in families for Np in NPS])
    is_2dfft = np.array([label.startswith("2D-FFT") for label, *_ in families for _ in NPS])
    is_flattop = np.array([label.endswith("flattop") for label, *_ in families for _ in NPS])

    # The 2D-FFT points share one execution time: label them in two columns, default on the left and
    # flattop on the right, spaced so that they do not overlap, with leader lines
    def spread(values, gap):
        order = np.argsort(values)
        placed = values[order].copy()
        for _ in range(50):
            for k in range(1, len(placed)):
                placed[k] = max(placed[k], placed[k - 1] + gap)
            placed -= np.mean(placed - values[order])
        result = np.empty_like(placed)
        result[order] = placed
        return result

    # Points that the other labels must avoid: every marker, plus the area each column label covers
    avoid_x, avoid_y = list(times), list(scores)
    for flattop, side in ((False, -1), (True, 1)):
        group = is_2dfft & (is_flattop == flattop)
        y_text = spread(scores[group], 0.0078)
        x_text = (times[group].min() / 2.2) if side < 0 else (times[group].max() * 1.45)
        for x, y, yt, text in zip(times[group], scores[group], y_text, labels[group]):
            ax.annotate(text, (x, y), xytext=(x_text, yt), ha="right" if side < 0 else "left", va="center",
                        fontsize=SMALL - 0.5, arrowprops=dict(arrowstyle="-", color=AXIS_GREY, linewidth=0.4, shrinkA=1, shrinkB=3))
            for step in np.linspace(0, 0.45, 6):  # the label and its leader line, in log2 units
                avoid_x.append(x_text * 2**(side * step))
                avoid_y.append(yt)
            avoid_x += list(np.geomspace(x, x_text, 4))
            avoid_y += list(np.linspace(y, yt, 4))
    others = ~is_2dfft
    ta.allocate(ax, times[others], scores[others], list(labels[others]), x_scatter=avoid_x, y_scatter=avoid_y,
                textsize=SMALL - 0.5, draw_lines=True, linewidth=0.4, linecolor=AXIS_GREY, margin=0.01, min_distance=0.01,
                max_distance=0.08, nbr_candidates=500, draw_all=True)

    ax.set_xlabel("Minimum execution time (seconds, log scale)", fontsize=LABEL, labelpad=1)
    ax.set_ylabel("Average AUC across SNR", fontsize=LABEL, labelpad=1)
    ax.tick_params(labelsize=TICK, length=2, pad=1.5)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=SMALL - 0.5, ncol=2, frameon=False, title="Numbers: $N_p$",
              title_fontsize=SMALL - 0.5, columnspacing=0.6, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.2,
              handlelength=1.3)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.98)
    print("Pareto-optimal:", ", ".join(f"{n} ({s:.3f})" for n, s, t in sorted(zip(np.array(names)[front], scores[front], times[front]), key=lambda z: z[2])))
    print("Fast-SC:", ", ".join(f"{Np}: {s:.3f}" for Np, s in zip(NPS, scores[4*len(NPS):5*len(NPS)])))
    return fig


if __name__ == "__main__":
    roc_figure().savefig(ROOT / "fig" / "roc_compare_subplot_rev_2.png", dpi=300)
    print("fig/roc_compare_subplot_rev_2.png written")
    if "--roc-only" not in sys.argv:
        pareto_figure().savefig(ROOT / "fig" / "pareto_front_rev.png", dpi=300)
        print("fig/pareto_front_rev.png written")
