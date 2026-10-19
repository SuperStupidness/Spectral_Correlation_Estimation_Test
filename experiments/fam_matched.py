from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
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


def diamond_vertices(f_c, a_c, f_half, a_half):
    return np.array([
        [f_c - f_half, a_c],
        [f_c, a_c + a_half],
        [f_c + f_half, a_c],
        [f_c, a_c - a_half],
    ])


def make_figures(Np=8, N=32, fs=1.0,
                 highlight_k=4, highlight_l=4):
    half_fs = fs / 2
    L = Np // 4
    P = N // L
    q_lim = int(P / 2)
    q_vals = np.arange(2 * q_lim) - q_lim

    delta_alpha = fs / N
    f_half_diag = fs / (2 * Np)
    a_half_diag = fs / Np

    def cell_width(q):
        return max(fs / Np - abs(q) / N, 0)

    # Compute all output points
    all_fj, all_alpha = [], []
    for k in range(Np):
        for l in range(Np):
            fj = (k + l - Np) / (2 * Np)
            for q in q_vals:
                a = (k - l) / Np + q / N
                all_fj.append(fj)
                all_alpha.append(a)
    all_fj = np.array(all_fj)
    all_alpha = np.array(all_alpha)

    def in_diamond(f, a):
        return np.abs(f) + np.abs(a) / 2 <= half_fs + 1e-10
    mask = in_diamond(all_fj, all_alpha)

    f_c = (highlight_k + highlight_l - Np) / (2 * Np)
    a_c = (highlight_k - highlight_l) / Np

    # Highlighted cell: pick one near top of diamond (narrow but nonzero)
    # Diamond closes at |q| = N/Np, so max useful |q| = N/Np - 1
    q_max_useful = int(N / Np) - 1
    q_hi = q_max_useful  # near tip, narrow cell

    print(f"Np={Np}, N={N}, L={L}, P={P}, q_lim={q_lim}")
    print(f"q_vals={q_vals}")
    print(f"Highlighted cell: q={q_hi}, delta_f={cell_width(q_hi):.4f}")

    # ================================================================
    # FIGURE (a): Full output grid
    # ================================================================
    f_lo, f_hi_lim = -half_fs - 0.04, half_fs + 0.04
    a_lo, a_hi_lim = -fs - 0.08, fs + 0.08
    f_range = f_hi_lim - f_lo
    a_range = a_hi_lim - a_lo
    data_ratio = a_range / f_range

    fig_a, ax_a = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    ax_a.plot(all_fj[mask], all_alpha[mask], 'x', color='#333333',
              ms=1.4, mew=0.4, zorder=4)

    # Highlighted CSA cell
    f_hi_cell = f_c
    a_hi_cell = a_c + q_hi / N
    w_hi = cell_width(q_hi)
    ax_a.add_patch(Rectangle((f_hi_cell - w_hi / 2, a_hi_cell - delta_alpha / 2),
                              w_hi, delta_alpha,
                              fc='#dd2222', ec='#aa0000', lw=0.8,
                              alpha=0.85, zorder=8))
    ax_a.plot(f_hi_cell, a_hi_cell, 'x', color='white', ms=2.0, mew=0.6, zorder=9)

    ax_a.annotate('CSA cell', xy=(f_hi_cell, a_hi_cell),
                  xytext=(f_hi_lim - 0.03, a_hi_cell - 0.3), ha='right',
                  fontsize=7, color='#aa0000', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#aa0000', lw=0.7,
                                  connectionstyle='arc3,rad=-0.2'),
                  bbox=dict(boxstyle='round,pad=0.15', fc='white',
                            ec='#aa0000', alpha=0.95, lw=0.5),
                  zorder=10)

    ax_a.axhline(0, color='gray', lw=0.3, ls='--', zorder=1)
    ax_a.axvline(0, color='gray', lw=0.3, ls='--', zorder=1)
    ax_a.set_xlabel('$f$ (Hz)')
    ax_a.set_ylabel('$\\alpha$ (Hz)')
    ax_a.set_xlim(f_lo, f_hi_lim)
    ax_a.set_ylim(a_lo, a_hi_lim)
    fig_a.subplots_adjust(**MARGINS)

    # ================================================================
    # FIGURE (b): Zoomed CSA cell
    # ================================================================
    zm_half_f = (fs / Np) * 1.0
    zm_half_a = zm_half_f * data_ratio

    fig_b, ax_b = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    # Background diamond outlines
    for k in range(Np):
        for l in range(Np):
            fc_n = (k + l - Np) / (2 * Np)
            ac_n = (k - l) / Np
            if (abs(fc_n - f_c) < zm_half_f * 1.5 and
                abs(ac_n - a_c) < zm_half_a * 1.5):
                dv = diamond_vertices(fc_n, ac_n, f_half_diag, a_half_diag)
                ax_b.add_patch(MplPolygon(dv, closed=True,
                                          fill=False, ec='#cccccc', ls='-',
                                          lw=0.4, zorder=1))

    # Highlighted diamond: all rectangular cells as outlines
    for q in q_vals:
        a_q = a_c + q / N
        w_q = cell_width(q)
        if w_q <= 0:
            continue
        is_selected = (q == q_hi)
        if is_selected:
            ax_b.add_patch(Rectangle(
                (f_c - w_q / 2, a_q - delta_alpha / 2),
                w_q, delta_alpha,
                fc='#dd2222', ec='#aa0000', lw=1.0,
                alpha=0.75, zorder=7))
        else:
            ax_b.add_patch(Rectangle(
                (f_c - w_q / 2, a_q - delta_alpha / 2),
                w_q, delta_alpha,
                fill=False, ec='#aa0000', lw=0.5, alpha=0.45, zorder=3))

    # Highlighted diamond outline (bold)
    dv_hi = diamond_vertices(f_c, a_c, f_half_diag, a_half_diag)
    ax_b.add_patch(MplPolygon(dv_hi, closed=True,
                               fill=False, ec='#aa0000', ls='-',
                               lw=1.0, alpha=0.8, zorder=6))

    # × markers — only highlighted cell is white
    for k in range(Np):
        for l in range(Np):
            fc_n = (k + l - Np) / (2 * Np)
            for q in q_vals:
                a_n = (k - l) / Np + q / N
                if (abs(fc_n - f_c) < zm_half_f * 1.1 and
                    abs(a_n - a_c) < zm_half_a * 1.1):
                    if (k == highlight_k and l == highlight_l and q == q_hi):
                        ax_b.plot(fc_n, a_n, 'x', color='white',
                                  ms=4, mew=0.8, zorder=9)
                    else:
                        ax_b.plot(fc_n, a_n, 'x', color='#333333',
                                  ms=4, mew=0.8, zorder=5)

    # --- Dimension labels ---
    ax_b.text(f_c - zm_half_f * 0.88, a_c + q_hi / N,
              '$\\Delta\\alpha = \\frac{f_s}{N}$',
              fontsize=6, color='#aa0000', fontweight='bold',
              ha='center', va='center',
              bbox=dict(fc='white', ec='#aa0000', alpha=0.95,
                        boxstyle='round,pad=0.15', lw=0.5),
              zorder=11)

    ax_b.text(f_c, a_c - a_half_diag - zm_half_a * 0.08,
              "$\\Delta f(q) = \\frac{f_s}{N_p} - \\frac{|q|}{N}$",
              fontsize=6, color='#aa0000', fontweight='bold',
              ha='center', va='top',
              bbox=dict(fc='white', ec='#aa0000', alpha=0.95,
                        boxstyle='round,pad=0.15', lw=0.5),
              zorder=11)

    ax_b.set_xlim(f_c - zm_half_f, f_c + zm_half_f)
    ax_b.set_ylim(a_c - zm_half_a, a_c + zm_half_a)
    ax_b.set_xlabel('$f$ (Hz)')
    ax_b.set_ylabel('$\\alpha$ (Hz)')
    # Short tick labels, so the axis label stays inside the figure
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    fig_b.subplots_adjust(**MARGINS)

    return fig_a, fig_b


if __name__ == '__main__':
    fig_a, fig_b = make_figures(Np=8, N=32, fs=1.0,
                                 highlight_k=4, highlight_l=4)
    fig_a.savefig(FIG_DIR / 'fam_output_grid.png', dpi=300)
    fig_b.savefig(FIG_DIR / 'fam_csa_cell.png', dpi=300)
    print("Done.")
