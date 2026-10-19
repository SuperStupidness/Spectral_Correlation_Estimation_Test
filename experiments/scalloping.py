"""Amplitude of a cyclic feature against its distance from the nearest cycle frequency bin.

    python experiments/scalloping.py

Explains the alternation in the Fast-SC validation RMSE between successive powers of two.
Changing the signal length N moves the BPSK feature at alpha = 0.1 relative to the 1/N bins.
At each length the alpha = 0.1 slice is extracted as validation_test does and fitted to the
theoretical SCD; the fitted amplitude is plotted against the bin offset, with the response
predicted by each estimator's window along time: rectangular for the Fast-SC, Hamming g(n) for
the SSCA and FAM, and boxcar g(n), also rectangular, for their flattop/boxcar configuration.

The Fast-SC (zero-padded) has a 1/N grid for any N, so it is swept finely from 8190 to 8200:
on a bin at 8190, half a bin off at 8195. The SSCA and FAM labels are on the 1/N grid only when
N is a multiple of Np (otherwise their strips are offset by fractions of a bin), so they use
multiples of 512, which reach offsets of 0, 0.2 and 0.4 bins; the Fast-SC is run there too.

Writes fig/scalloping.png and .pdf, and data/revision/scalloping.csv. With --plot-only, redraws
the figure from that file.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scf_test.tests as test
from scf_test.algorithm import ssca, fam, ssca_flattop, fam_flattop, fast_sc_wrapper
from scf_test.psk import create_rect_bpsk_signal

NP, SAMPLES_PER_SYMBOL, CFO, SNR, ALPHA = 512, 10, 0.05, 10, 0.1
FINE_LENGTHS = range(8190, 8201)  # Fast-SC only
GRID_LENGTHS = range(8192, 11265, 512)  # multiples of Np, every estimator
REALIZATIONS = 3
ESTIMATORS = {
    "Fast-SC": lambda x: fast_sc_wrapper(x, Np=NP, alpha_max=0.5),
    "SSCA": lambda x: ssca(x, Np=NP),
    "FAM": lambda x: fam(x, Np=NP, L=NP // 4),
    "SSCA flattop/boxcar": lambda x: ssca_flattop(x, Np=NP),
    "FAM flattop/boxcar": lambda x: fam_flattop(x, Np=NP, L=NP // 8),
}


def window_response(window, offset, size=8192):
    """Amplitude of the window's transform `offset` bins from its centre, relative to the centre."""
    w = np.ones(size) if window == "rectangular" else np.hamming(size)
    return abs(np.sum(w * np.exp(-2j*np.pi*offset*np.arange(size)/size))) / w.sum()


def amplitude(task):
    """Least-squares amplitude of the alpha = 0.1 slice relative to theory, for one estimator, length and realization."""
    name, N, r = task
    rng = np.random.default_rng([2026, N, r])
    signal = test.add_cfo(create_rect_bpsk_signal(N, SAMPLES_PER_SYMBOL, rng=rng), CFO)
    signal = test.add_awgn_snr(signal, desired_snr=SNR, rng=rng)[:N]

    scd, f, alpha = ESTIMATORS[name](signal)
    mask = (np.abs(np.round(alpha*N)/N - ALPHA) <= 1/(2*N)) & (np.abs(f) + np.abs(alpha/2) < 0.5)
    estimate = np.abs(scd[mask])
    reference = np.abs(test.bpsk_scf_signed(f[mask], ALPHA, SAMPLES_PER_SYMBOL, CFO))
    offset = round(abs(ALPHA*N - np.rint(ALPHA*N)), 3)
    return name, N, r, offset, np.sum(estimate*reference) / np.sum(reference**2)


if __name__ == "__main__":
    path = ROOT / "data" / "revision" / "scalloping.csv"
    if "--plot-only" in sys.argv:
        table = pd.read_csv(path)
    else:
        tasks = [(name, N, r) for name in ESTIMATORS for N in GRID_LENGTHS for r in range(REALIZATIONS)]
        tasks += [("Fast-SC", N, r) for N in FINE_LENGTHS if N not in GRID_LENGTHS for r in range(REALIZATIONS)]
        with ProcessPoolExecutor(8) as pool:
            rows = list(pool.map(amplitude, tasks))
        table = pd.DataFrame(rows, columns=["estimator", "N", "realization", "bin_offset", "amplitude"])
        table.to_csv(path, index=False)

    means = table.groupby(["estimator", "bin_offset"])["amplitude"].mean().unstack(0)
    means["rectangular predicts"] = [window_response("rectangular", d) for d in means.index]
    means["Hamming predicts"] = [window_response("hamming", d) for d in means.index]
    print(means.round(3).to_string())

    # One colour per configuration (SSCA, FAM and Fast-SC as in the validation figure); the predicted
    # curves are neutral. Filled markers have a rectangular window along time, hollow ones Hamming.
    # Markers are nudged sideways so they do not overlap.
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    grid = np.linspace(0, 0.5, 101)
    ax.plot(grid, [window_response("rectangular", d) for d in grid], color="black", lw=1, label="Rectangular window")
    ax.plot(grid, [window_response("hamming", d) for d in grid], color="grey", lw=1, ls="--", label="Hamming window")
    for name, color, marker, nudge, filled in (("Fast-SC", "tab:green", "^", -0.012, True),
                                               ("SSCA flattop/boxcar", "tab:purple", "D", 0.0, True),
                                               ("FAM flattop/boxcar", "tab:red", "s", 0.012, True),
                                               ("SSCA", "tab:blue", "o", -0.006, False),
                                               ("FAM", "tab:orange", "v", 0.006, False)):
        stats = table[table["estimator"] == name].groupby("bin_offset")["amplitude"].agg(["mean", "std"])
        ax.errorbar(stats.index + nudge, stats["mean"], yerr=stats["std"], fmt=marker, color=color, ms=4,
                    mfc=color if filled else "white", elinewidth=0.7, capsize=1.5, label=name)
    ax.set_xlabel("Offset of the cycle frequency (bins)")
    ax.set_ylabel("Amplitude / theory")
    ax.set_xlim(-0.03, 0.53)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / "fig" / f"scalloping.{extension}", dpi=300, bbox_inches="tight")
    print("fig/scalloping.png and .pdf written")
