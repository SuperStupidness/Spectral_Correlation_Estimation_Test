"""Robustness test AUC for each signal, default against flattop windows, at each signal length (0 dB, Np = 64).

    python experiments/robustness_breakdown.py

Reads the stored statistics in data/revision/roc (500 trials per signal) and writes fig/supp_robustness_signals
(.pdf and .png) and supplementary/supp_robustness_signals.pdf. Also prints the AUC table.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from revision_summary import auc
from revision_figures import CM, TEXT_WIDTH, ESTIMATORS, MUTED, style  # shared style of the supplement figures

LENGTHS = (2**10, 2**12, 2**14, 2**16)
CONFIGS = [("ssca", False), ("ssca", True), ("fam", False), ("fam", True)]
LABELS = {"bpsk": "BPSK", "qpsk": "QPSK", "gmsk": "GMSK", "msk": "MSK", "srrc_bpsk": "SRRC BPSK"}


def per_signal_auc(estimator, flattop, N, conjugate):
    name = f"roc_{estimator}{'_flattop' if flattop else ''}_Np64_N{N}_snr0_{'c' if conjugate else 'nc'}.npz"
    data = np.load(ROOT / "data" / "revision" / "roc" / name)
    return [str(s) for s in data["signals"]], [auc(present, absent) for present, absent in data["statistic"]]


def figure():
    fig, axes = plt.subplots(2, len(LENGTHS), figsize=(TEXT_WIDTH, 6.8 * CM), sharey=True)
    width = 0.2
    for row, conjugate in enumerate((False, True)):
        for column, N in enumerate(LENGTHS):
            ax = axes[row, column]
            for k, (estimator, flattop) in enumerate(CONFIGS):
                signals, values = per_signal_auc(estimator, flattop, N, conjugate)
                color = ESTIMATORS[estimator][1]
                x = np.arange(len(signals)) + (k - 1.5) * width
                ax.bar(x, values, width * 0.9, color="white" if flattop else color, edgecolor=color,
                       hatch="////" if flattop else None, linewidth=0.8)
            ax.set_xticks(np.arange(len(signals)), [LABELS[s] for s in signals])
            ax.set_ylim(0.4, 1.02)
            ax.axhline(0.5, color=MUTED, linewidth=0.5, linestyle=":")
            exponent = int(np.log2(N))
            ax.set_title(f"{'Conjugate' if conjugate else 'Non-conjugate'}, $N = 2^{{{exponent}}}$", loc="left")
            style(ax)
            ax.grid(axis="x", visible=False)
        axes[row, 0].set_ylabel("AUC")
    keys = [Patch(facecolor=ESTIMATORS[e][1], edgecolor=ESTIMATORS[e][1], label=ESTIMATORS[e][0]) for e in ("ssca", "fam")]
    keys += [Patch(facecolor="white", edgecolor=ESTIMATORS[e][1], hatch="////", label=f"{ESTIMATORS[e][0]}, flattop")
             for e in ("ssca", "fam")]
    fig.legend(handles=keys, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.84, wspace=0.08, hspace=0.5)
    return fig


if __name__ == "__main__":
    for conjugate in (False, True):
        print("conjugate" if conjugate else "non-conjugate")
        for N in LENGTHS:
            for estimator, flattop in CONFIGS:
                signals, values = per_signal_auc(estimator, flattop, N, conjugate)
                print(f"  N = 2^{int(np.log2(N)):2d}  {estimator + (' flattop' if flattop else ''):13s} "
                      + "  ".join(f"{s} {v:.3f}" for s, v in zip(signals, values)))
    fig = figure()
    for path in (ROOT / "fig" / "supp_robustness_signals.pdf", ROOT / "supplementary" / "supp_robustness_signals.pdf"):
        fig.savefig(path)
    fig.savefig(ROOT / "fig" / "supp_robustness_signals.png", dpi=300)
    print("fig/supp_robustness_signals.pdf and .png written")
