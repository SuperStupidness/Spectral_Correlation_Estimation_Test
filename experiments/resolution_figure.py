"""Two-tone resolution test: the alpha = 0 slice of each estimator for two tones a few bins apart.

    python experiments/resolution_figure.py              # compute and plot
    python experiments/resolution_figure.py --plot-only  # redraw from data/revision/resolution.npz

Two equal complex tones, spaced Delta bins of 1/Np apart, are slid together across one grid spacing in
small steps. A shift of the input by delta moves the output by delta, so interleaving the outputs gives the
alpha = 0 slice on a fine frequency grid even where the estimator's own grid is 1/Np. The slice is
normalized to its largest value; the dip between the tones measures whether they are resolved.
N = 4096, Np = 64, noise free, SCD (not coherence).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

N, NP = 4096, 64
CENTRE = 0.2                                        # inside Fast-SC's [0, fs/2]
SEPARATIONS = np.arange(1, 8.01, 0.5)               # in bins of 1/Np
SHIFTS = 16                                         # steps across one 1/Np grid spacing
OUT = ROOT / "data" / "revision" / "resolution.npz"
PANEL_SEPARATIONS = (2, 3, 4)
CONFIGS = ["ssca", "fam", "2dfft", "fastsc", "ssca_flattop", "fam_flattop", "2dfft_flattop"]
LABELS = {"ssca": "SSCA", "fam": "FAM", "2dfft": "2D-FFT", "fastsc": "Fast-SC", "ssca_flattop": "SSCA flattop",
          "fam_flattop": "FAM flattop", "2dfft_flattop": "2D-FFT flattop"}
# The SSCA and 2D-FFT curves coincide, so the SSCA is drawn wide and pale underneath
STYLE = {"ssca": dict(color="tab:blue", linewidth=3, alpha=0.4), "fam": dict(color="tab:red"),
         "2dfft": dict(color="tab:pink"), "fastsc": dict(color="tab:brown"),
         "ssca_flattop": dict(color="tab:green", linewidth=3, alpha=0.4), "fam_flattop": dict(color="tab:orange", linestyle="--"),
         "2dfft_flattop": dict(color="tab:purple", linestyle="--")}


def style(name):
    return {"linewidth": 0.9, **STYLE[name]}


def alpha_zero_slice(name, x):
    from revision_runs import get_estimator
    func, kwargs = get_estimator(name, NP)
    S, f, a = func(x, Np=NP, **kwargs)
    S, f, a = (np.asarray(v).ravel() for v in (S, f, a))
    zero = np.abs(a) < 0.5 / N
    return f[zero], np.abs(S[zero])


def two_tone_slice(name, separation):
    """Fine-grid alpha = 0 slice for two tones `separation` bins apart, centred on CENTRE."""
    n = np.arange(N)
    steps = 1 if name.startswith("2dfft") else SHIFTS          # the 2D-FFT slice is already on a 1/N grid
    fs, ss = [], []
    for d in np.arange(steps) / (steps * NP):
        f1, f2 = CENTRE + d - separation / (2 * NP), CENTRE + d + separation / (2 * NP)
        x = np.exp(2j * np.pi * f1 * n) + np.exp(2j * np.pi * f2 * n + 1j)
        f, s = alpha_zero_slice(name, x)
        keep = np.abs(f - CENTRE - d) < 8 / NP
        fs.append((f[keep] - d - CENTRE) * NP)                   # in bins, relative to the pair's centre
        ss.append(s[keep])
    f, s = np.concatenate(fs), np.concatenate(ss)
    order = np.argsort(f)
    return f[order], s[order] / s.max()


def compute():
    data = {}
    for name in CONFIGS:
        dips = []
        for sep in SEPARATIONS:
            f, s = two_tone_slice(name, sep)
            dips.append(np.interp(0, f, s))                      # value midway between the tones
            if sep in PANEL_SEPARATIONS:
                data[f"{name}_f_{sep:g}"], data[f"{name}_s_{sep:g}"] = f, s
        data[f"{name}_dip"] = np.array(dips)
        print(name, "separation where the midpoint first falls below 0.5:",
              next((f"{s:g} bins" for s, d in zip(SEPARATIONS, dips) if d < 0.5), "> 8 bins"), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, separations=SEPARATIONS, **data)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    fig, axes = plt.subplots(1, len(PANEL_SEPARATIONS) + 1, figsize=(7.16, 1.6),
                             gridspec_kw={"width_ratios": [1] * len(PANEL_SEPARATIONS) + [1.25]})
    for ax, sep in zip(axes, PANEL_SEPARATIONS):
        for name in CONFIGS:
            ax.plot(d[f"{name}_f_{sep:g}"], d[f"{name}_s_{sep:g}"], **style(name))
        for t in (-sep / 2, sep / 2):
            ax.axvline(t, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
        ax.set(xlim=(-5, 5), ylim=(0, 1.05), title=f"Tones {sep:g} bins apart")
        ax.set_xlabel(r"$(f - f_c) N_p$")
    axes[0].set_ylabel("Normalized SCD")
    ax = axes[-1]
    for name in CONFIGS:
        ax.plot(d["separations"], d[f"{name}_dip"], label=LABELS[name], **style(name))
    ax.axhline(0.5, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
    ax.set(xlim=(1, 8), ylim=(0, 1.05), title="Midpoint value")
    ax.set_xlabel(r"Tone spacing $\times N_p$")
    for a in axes:
        a.tick_params(labelsize=7)
        a.title.set_fontsize(8)
        a.xaxis.label.set_fontsize(8)
        a.yaxis.label.set_fontsize(8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for path in (ROOT / "fig" / "supp_resolution.png", ROOT / "fig" / "supp_resolution.pdf",
                 ROOT / "supplementary" / "supp_resolution.pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print("fig/supp_resolution.png and supplementary/supp_resolution.pdf written")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        compute()
    plot()
