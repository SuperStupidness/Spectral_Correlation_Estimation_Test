"""Cycle frequency identification figure of the paper: cyclic domain profile of one BPSK realization.

    python experiments/cdp_figure.py
    python experiments/cdp_figure.py --full --name fig4_coherence_rev_thick
    python experiments/cdp_figure.py --snr 0 --name fig4_coherence_rev_0dB

Rectangular BPSK, N = 4096, Np = 64, symbol rate 0.1, carrier offset 0.05. The
Fast-SC uses its own coherence option, as in the paper; pass --fastsc-welch to
normalize it by the Welch PSD like the other estimators. Same layout as the
original figure drawn in Examples.ipynb: alpha >= 0 (the non-conjugate CDP is
symmetric; --full for -0.95 to 0.95), grey bands at the significant cycle frequencies
k f_sym, and the noisiest profile drawn first so that the others stay visible on top.
The S3CA's sparse FFT is randomized, so its profile changes from run to run even with
the same seed; the other estimators reproduce exactly. Writes .png and .pdf to fig/.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

import scf_test.tests as test
from scf_test.algorithm import ssca, fam, fast_sc_wrapper, scf_2d_fft_wrapper
from scf_test.psk import create_rect_bpsk_signal
from algorithm.s3ca import s3ca

N, NP, SAMPLES_PER_SYMBOL, CFO = 4096, 64, 10, 0.05
ALPHA_RANGE = (-0.95, 0.95)


def cyclic_domain_profile(coherence, alpha):
    """max over f of |coherence| at each cycle frequency, divided by its value at alpha = 0."""
    magnitude = np.abs(np.asarray(coherence)).ravel()
    alpha_unique, inverse = np.unique(np.round(np.asarray(alpha).ravel(), decimals=10), return_inverse=True)
    profile = np.zeros(len(alpha_unique))
    np.maximum.at(profile, inverse, magnitude)
    return alpha_unique, profile / profile[np.argmin(np.abs(alpha_unique))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snr", type=float, default=10, help="SNR in dB")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fastsc-welch", action="store_true", help="normalize the Fast-SC by the Welch PSD")
    parser.add_argument("--full", action="store_true", help="show -0.95 <= alpha <= 0.95 instead of alpha >= 0")
    parser.add_argument("--name", default="fig4_coherence_rev_thick_half", help="file name inside fig/")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    signal = test.add_cfo(create_rect_bpsk_signal(N, SAMPLES_PER_SYMBOL, rng=rng), CFO)
    signal = test.add_awgn_snr(signal, desired_snr=args.snr, rng=rng)[:N]

    # Drawn in this order: the noisiest floor first, so the others stay visible on top of it
    dense = {
        "FAM": (fam(signal, Np=NP, L=NP // 4, coherence=True), "tab:orange"),
        "SSCA": (ssca(signal, Np=NP, coherence=True), "tab:blue"),
        "Fast-SC": (fast_sc_wrapper(signal, Np=NP, coherence=True, builtin_coherence=not args.fastsc_welch),
                    "tab:green"),
        "2D-FFT": (scf_2d_fft_wrapper(signal, Np=NP, coherence=True), "tab:red"),
    }
    sparse = {"S3CA": (s3ca(signal, Np=NP, coherence=True, seed=23), "0.35")}

    low = ALPHA_RANGE[0] if args.full else 0
    fig, axes = plt.subplots(2, 1, figsize=(6, 3), sharex=True, gridspec_kw={"hspace": 0.15})
    theory = np.arange(-9, 10) / SAMPLES_PER_SYMBOL
    for ax, estimates in ((axes[0], dense), (axes[1], sparse)):
        for a in (theory if args.full else theory[theory > 0]):
            ax.axvline(a, color="0.85", linewidth=3, zorder=0)
        for label, ((coherence, _, alpha), color) in estimates.items():
            alpha_unique, profile = cyclic_domain_profile(coherence, alpha)
            shown = (alpha_unique >= low) & (alpha_unique <= ALPHA_RANGE[1])
            ax.plot(alpha_unique[shown], profile[shown], color=color, label=label, linewidth=1.1)
            peaks = [profile[np.argmin(np.abs(alpha_unique - k/SAMPLES_PER_SYMBOL))] for k in (1, 2, 3, 4)]
            inside = (np.abs(alpha_unique) > 0.02) & (np.abs(alpha_unique) < 0.48)
            print(f"{label:8s} peaks at alpha = 0.1 to 0.4: " + ", ".join(f"{p:.2f}" for p in peaks)
                  + f"; median elsewhere {np.median(profile[inside]):.2f}")
        ax.set_xlim(low, ALPHA_RANGE[1])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel(r"Cycle frequency $\alpha$")
    fig.supylabel("Norm. |C(f, α)|", fontsize=10)

    order = ["SSCA", "FAM", "2D-FFT", "Fast-SC", "S3CA"]
    pairs = dict(pair for ax in axes for pair in zip(*reversed(ax.get_legend_handles_labels())))
    fig.legend([pairs[k] for k in order], order, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.98),
               fontsize=10, frameon=False, handlelength=1.5)
    fig.subplots_adjust(top=0.88, bottom=0.12)

    for extension in ("png", "pdf"):
        fig.savefig(ROOT / "fig" / f"{args.name}.{extension}", bbox_inches="tight", dpi=300)
    print(f"fig/{args.name}.png and .pdf written")


if __name__ == "__main__":
    main()
