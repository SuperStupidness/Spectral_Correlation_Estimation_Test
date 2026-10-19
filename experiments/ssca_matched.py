from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rcParams.update({
    'axes.labelpad': 2,
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

CM = 1/2.54
FIG_SIZE = 3.5 * CM  # square panel; include at this width in LaTeX to keep the text size
MARGINS = dict(left=0.38, right=0.97, bottom=0.29, top=0.97)
FIG_DIR = Path(__file__).resolve().parents[1] / 'fig'

def make_figures(Np=8, N=32, fs=1.0, highlight_k=0, highlight_q=2):
    half_fs = fs / 2
    delta_f = fs / Np
    delta_a = fs / N

    k_vals = np.arange(-Np//2, Np//2)
    q_vals = np.arange(-N//2, N//2)
    k_grid = k_vals.reshape(-1, 1)
    q_grid = q_vals.reshape(1, -1)
    f_all = k_grid / (2*Np) - q_grid / (2*N)
    alpha_all = k_grid / Np + q_grid / N

    f_c = highlight_k/(2*Np) - highlight_q/(2*N)
    a_c = highlight_k/Np + highlight_q/N

    f_pad = 0.06
    a_pad = 0.12
    f_lo, f_hi = -half_fs - f_pad, half_fs + f_pad
    a_lo, a_hi = -fs - a_pad, fs + a_pad
    main_f_range = f_hi - f_lo
    main_a_range = a_hi - a_lo
    data_ratio = main_a_range / main_f_range

    # ================================================================
    # FIGURE (a): Full output grid
    # ================================================================
    fig_a, ax_a = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    for idx in range(Np):
        fp = f_all[idx, :]
        ap = alpha_all[idx, :]
        ax_a.plot(fp, ap, 'x', color='#333333', ms=1.6, mew=0.4, zorder=4)

    ax_a.add_patch(Rectangle((f_c - delta_f/2, a_c - delta_a/2),
                              delta_f, delta_a,
                              fc='#dd2222', ec='#aa0000', lw=0.8,
                              alpha=0.8, zorder=8))
    ax_a.plot(f_c, a_c, 'x', color='white', ms=2.0, mew=0.6, zorder=9)

    ax_a.axhline(0, color='gray', lw=0.3, ls='--', zorder=1)
    ax_a.axvline(0, color='gray', lw=0.3, ls='--', zorder=1)
    ax_a.set_xlabel('$f$ (Hz)')
    ax_a.set_ylabel('$\\alpha$ (Hz)')
    ax_a.set_xlim(f_lo, f_hi)
    ax_a.set_ylim(a_lo, a_hi)

    ax_a.annotate('CSA cell', xy=(f_c, a_c),
                  xytext=(f_hi - 0.03, a_c + 0.5), ha='right',
                  fontsize=7, color='#aa0000', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#aa0000', lw=0.7,
                                  connectionstyle='arc3,rad=0.2'),
                  bbox=dict(boxstyle='round,pad=0.15', fc='white',
                            ec='#aa0000', alpha=0.95, lw=0.5),
                  zorder=10)

    fig_a.subplots_adjust(**MARGINS)

    # ================================================================
    # FIGURE (b): Zoomed CSA cell
    # ================================================================
    zm_half_f = delta_f * 1.8
    zm_half_a = zm_half_f * data_ratio

    fig_b, ax_b = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    for idx in range(Np):
        fp = f_all[idx, :]
        ap = alpha_all[idx, :]
        mask = ((fp > f_c - zm_half_f*1.2) & (fp < f_c + zm_half_f*1.2) &
                (ap > a_c - zm_half_a*1.2) & (ap < a_c + zm_half_a*1.2))
        ax_b.plot(fp[mask], ap[mask], 'x', color='#333333',
                  ms=4, mew=0.8, zorder=5)

    ax_b.add_patch(Rectangle((f_c - delta_f/2, a_c - delta_a/2),
                              delta_f, delta_a,
                              fc='#dd2222', ec='#aa0000', lw=1.2,
                              alpha=0.7, zorder=7))
    ax_b.plot(f_c, a_c, 'x', color='white', ms=4.5, mew=1.1, zorder=8)

    ax_b.set_xlim(f_c - zm_half_f, f_c + zm_half_f)
    ax_b.set_ylim(a_c - zm_half_a, a_c + zm_half_a)
    ax_b.set_xlabel('$f$ (Hz)')
    ax_b.set_ylabel('$\\alpha$ (Hz)')

    # Short tick labels, so the axis label stays inside the figure
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))

    ax_b.text(f_c, a_c - delta_a/2 - zm_half_a * 0.18,
              "$\\Delta f = \\frac{f_s}{N_p}$",
              fontsize=7, color='#aa0000', fontweight='bold',
              bbox=dict(fc='white', ec='#aa0000', alpha=0.95,
                        boxstyle='round,pad=0.2', lw=0.5),
              ha='center', va='top', zorder=11)

    ax_b.text(f_c, a_c + delta_a/2 + zm_half_a * 0.18,
              '$\\Delta\\alpha = \\frac{f_s}{N}$',
              fontsize=7, color='#aa0000', fontweight='bold',
              bbox=dict(fc='white', ec='#aa0000', alpha=0.95,
                        boxstyle='round,pad=0.2', lw=0.5),
              ha='center', va='bottom', zorder=11)

    fig_b.subplots_adjust(**MARGINS)

    return fig_a, fig_b


if __name__ == '__main__':
    fig_a, fig_b = make_figures(Np=8, N=32, fs=1.0, highlight_k=0, highlight_q=2)
    fig_a.savefig(FIG_DIR / 'ssca_output_grid.png', dpi=300)
    fig_b.savefig(FIG_DIR / 'ssca_csa_cell.png', dpi=300)
    print("Done.")
