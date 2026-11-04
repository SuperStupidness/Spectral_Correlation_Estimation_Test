"""Pareto fronts at N = 1024, 4096, 8192 and 16384, one figure each (README; the paper's Fig. 7 is the N = 4096 panel).

    python experiments/revision_runs.py --pareto-n          # robustness runs at N = 1024, 8192 and 16384
    PARETO_N=1024 python -m pytest experiments/pareto_benchmark.py ... --benchmark-json=data/revision/pareto_benchmark_N1024.json
    PARETO_N=16384 python -m pytest experiments/pareto_benchmark.py ... --benchmark-json=data/revision/pareto_benchmark_N16384.json
    PARETO_N=8192 ...  (same, writing pareto_benchmark_N8192.json)
    python experiments/pareto_n_figure.py

Scores and times as in Fig. 7 (roc_figures.py). The 2D-FFT is absent above N = 4096 (O(N^2) memory) and the
Fast-SC is there only for Np <= 64. Writes fig/pareto_front_N1024.png, _N4096.png, _N8192.png and _N16384.png.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roc_figures import ROOT, NPS, pareto_score, benchmark_minima, find_pareto_front

LENGTHS = [1024, 4096, 8192, 16384]
# Configuration, label, colour and marker as in Figs. 6 and 7
FAMILIES = [("ssca", "SSCA", "blue", "x"), ("fam", "FAM", "red", "o"), ("ssca_flattop", "SSCA flattop", "green", "^"),
            ("fam_flattop", "FAM flattop", "orange", "s"), ("fastsc", "FastSC", "brown", "."),
            ("2dfft", "2D-FFT", "pink", "*"), ("2dfft_flattop", "2D-FFT flattop", "purple", "+")]


def points(N):
    """(configuration, Np, score, time) for every configuration run at this N."""
    minima = benchmark_minima(N)
    result = []
    for name, *_ in FAMILIES:
        for Np in NPS:
            if (name, Np) in minima:
                result.append((name, Np, pareto_score(name, Np, N), minima[(name, Np)]))
    return result


def main():
    for N in LENGTHS:
        fig, ax = plt.subplots(figsize=(9, 6))
        data = points(N)
        scores = np.array([s for *_, s, _ in data])
        times = np.array([t for *_, t in data])
        front = find_pareto_front(scores, times)
        order = np.argsort(times[front])
        ax.plot(times[front][order], scores[front][order], color="black", linestyle="--", linewidth=0.9,
                label="Pareto front", zorder=1)
        ax.scatter(times[front], scores[front], s=70, facecolors="none", edgecolors="black", linewidths=0.8, zorder=4)
        for name, label, color, marker in FAMILIES:
            mine = [i for i, (n, *_) in enumerate(data) if n == name]
            if mine:
                ax.plot(times[mine], scores[mine], color=color, linewidth=0.7, alpha=0.5, zorder=2)
                ax.plot(times[mine], scores[mine], linestyle="none", marker=marker, ms=5, color=color, label=label,
                        zorder=3)
        for i in range(len(data)):  # Np of every point
            ax.annotate(str(data[i][1]), (times[i], scores[i]), xytext=(4, -10), textcoords="offset points",
                        fontsize=8, color="0.3")
        ax.set_xscale("log", base=2)
        ax.set_title(f"Pareto front, N = {N}", fontsize=13)
        ax.set_xlabel("Minimum execution time (s, log scale)", fontsize=11)
        ax.set_ylabel("Average AUC across SNR", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=9, title="Numbers: $N_p$", title_fontsize=9)
        fig.tight_layout()
        fig.savefig(ROOT / "fig" / f"pareto_front_N{N}.png", dpi=200)
        plt.close(fig)
        print(f"N = {N}: " + ", ".join(f"{data[i][0]}_{data[i][1]} ({scores[i]:.3f}, {times[i]*1e3:.2f} ms)"
                                       for i in np.flatnonzero(front)[np.argsort(times[front])]))
    print("fig/pareto_front_N<N>.png written")


if __name__ == "__main__":
    main()
