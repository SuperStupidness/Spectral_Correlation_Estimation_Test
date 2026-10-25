"""Supplementary Figs. 8-12, from the results of revision_runs.py.

    python experiments/revision_summary.py     # writes the tables this script reads
    python experiments/revision_figures.py

Writes supp_sensitivity (Fig. 8), supp_convergence (Fig. 9), supp_fam_aliasing (Fig. 10), supp_scalloping
(Fig. 11, needs scalloping.py first) and supp_cdp_sensitivity (Fig. 12) to fig/, as .pdf for LaTeX and .png
to look at. supp_convergence is one column wide, the others the full text width.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import t as student_t

from revision_summary import load, pool_batches

matplotlib.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.1,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.labelpad': 2,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'revision'
FIG_DIR = ROOT / 'fig'
CM = 1/2.54
TEXT_WIDTH = 18.1 * CM  # two columns of the IEEE layout
COLUMN_WIDTH = 8.9 * CM

GRID = '#e1e0d9'
AXIS = '#898781'
MUTED = '#52514e'
BLUE, BLUE_LIGHT, BLUE_DARK = '#2a78d6', '#86b6ef', '#1c5cab'

# One hue and one marker per estimator; the flattop variants reuse both, dashed and hollow
ESTIMATORS = {
    'ssca': ('SSCA', '#2a78d6', 'o'),
    'fam': ('FAM', '#eb6834', 's'),
    '2dfft': ('2D-FFT', '#1baf7a', '^'),
    'fastsc': ('Fast-SC', '#eda100', 'D'),
    's3ca': ('S3CA', '#e87ba4', 'v'),
}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(AXIS)
    ax.tick_params(colors=MUTED, which='both')
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def plain_log_axis(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))


def half_width_percent(values):
    """95% half-width of the mean of the first n values, as a percentage of that mean."""
    counts = np.arange(2, len(values) + 1)
    half = [student_t.ppf(0.975, n - 1) * values[:n].std(ddof=1) / np.sqrt(n) / values[:n].mean() * 100 for n in counts]
    return counts, np.array(half)


def band(ax, x, curves, color, name, dashed=False):
    """Range over the configurations as a wash, their median as the line, named with its end value."""
    curves = np.array(curves)
    middle = np.median(curves, axis=0)
    ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15, linewidth=0)
    ax.plot(x, middle, color=color, linestyle='--' if dashed else '-',
            label=f'{name}: {middle[-1]:.1f}% at {x[-1]}')
    ax.plot(x[-1], middle[-1], 'o', color=color, ms=4, mec='white', mew=0.7, zorder=5)


def convergence_figure():
    """How the uncertainty of each test's metric shrinks with the number of trials (N = 4096, Np = 64).

    One column wide. Each curve ends at the number of trials used in the paper.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.3 * CM))
    blue, orange, aqua = (ESTIMATORS[name][1] for name in ('ssca', 'fam', '2dfft'))

    # Robustness test: every estimator, SCD type and SNR at the paper's setting
    roc = pd.read_csv(DATA / 'summary' / 'roc_auc_convergence.csv')
    roc['half'] = (roc['ci95_high'] - roc['ci95_low']) / 2 / roc['mean_auc'] * 100
    configs = roc.pivot_table(index=['estimator', 'scd', 'snr_db'], columns='trials', values='half')
    band(ax, configs.columns.to_numpy(), configs.to_numpy(), blue, 'Robustness, AUC')

    # Cycle leakage test: every estimator and SCD type at 0 dB, apart from the heavy-tailed S3CA
    leakage = [job for job in load(DATA / 'leakage')
               if job['N'] == 4096 and job['Np'] == 64 and job['snr'] == 0 and job['est'] != 's3ca']
    counts, _ = half_width_percent(leakage[0]['per_run'])
    keep = counts >= 5
    band(ax, counts[keep], [half_width_percent(job['per_run'])[1][keep] for job in leakage], orange, 'Cycle leakage, mean')

    # Validation test: SSCA and FAM (both SCD types) and the Fast-SC, at the shortest and longest signal
    # (RMSE of the complex values, as in the paper's validation figure)
    validation = pool_batches(load(DATA / 'validation'))
    for index, name, dashed in ((0, '$N = 2^{10}$', True), (-1, '$N = 2^{19}$', False)):
        counts, _ = half_width_percent(validation[0]['rmse_complex'][:, index])
        keep = counts >= 3
        band(ax, counts[keep], [half_width_percent(job['rmse_complex'][:, index])[1][keep] for job in validation], aqua,
             f'Validation, RMSE, {name}', dashed)

    ax.set(xscale='log', yscale='log', xlim=(2.6, 1300), ylim=(0.18, 400),
           xlabel='Number of trials', ylabel='95% half-width (%)')
    plain_log_axis(ax.xaxis, [3, 5, 10, 20, 50, 100, 200, 500, 1000])
    plain_log_axis(ax.yaxis, [0.3, 1, 3, 10, 30, 100])
    style(ax)
    ax.legend(loc='upper right', frameon=False, handlelength=2.2, borderaxespad=0.2, labelspacing=0.25)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.2, top=0.97)
    return fig


def series(ax, x, y, estimator, flattop=False):
    _, color, marker = ESTIMATORS[estimator]
    ax.plot(x, y, color=color, marker=marker, ms=4.5, mew=0.9,
            linestyle='--' if flattop else '-',
            mfc='white' if flattop else color, mec=color if flattop else 'white')


def sensitivity_figure():
    """Cycle leakage and robustness against the signal length N (top row, Np = 64) and the channel count Np
    (bottom row, N = 4096), at 0 dB. Columns: median cycle leakage, non-conjugate AUC, conjugate AUC."""
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 6.6 * CM))
    (leak_n, nc_n, c_n), (leak_np, nc_np, c_np) = axes

    def power_of_two_axis(ax):
        ax.xaxis.set_major_locator(FixedLocator([2**10, 2**12, 2**14, 2**16]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'$2^{{{int(np.log2(value))}}}$'))
        ax.xaxis.set_minor_locator(NullLocator())

    # Median leakage on log axes, so the flattop configuration's constant factor shows as a constant gap.
    # S3CA is left out: its median is four to eight times higher and would flatten the rest.
    leakage = pd.read_csv(DATA / 'summary' / 'leakage.csv')
    leakage = leakage[(leakage['snr_db'] == 0) & (leakage['scd'] == 'non-conjugate')]

    def leakage_lines(ax, rows, column):
        for name in rows['estimator'].unique():
            estimator, flattop = name.removesuffix('_flattop'), name.endswith('_flattop')
            if estimator == 's3ca':
                continue
            line = rows[rows['estimator'] == name].sort_values(column)
            series(ax, line[column], line['median'], estimator, flattop)
        ax.set(xscale='log', yscale='log', ylabel='Median leakage')
        plain_log_axis(ax.yaxis, [0.02, 0.05, 0.1, 0.2, 0.5])

    leakage_lines(leak_n, leakage[leakage['Np'] == 64], 'N')
    leakage_lines(leak_np, leakage[leakage['N'] == 4096], 'Np')

    roc = pd.read_csv(DATA / 'summary' / 'roc_auc.csv')
    roc = roc[roc['snr_db'] == 0]

    def auc_lines(ax, rows, column):
        for name in rows['estimator'].unique():
            estimator, flattop = name.removesuffix('_flattop'), name.endswith('_flattop')
            if estimator not in ESTIMATORS:  # variants such as the Fast-SC with the Welch normalization
                continue
            line = rows[rows['estimator'] == name].sort_values(column)
            series(ax, line[column], line['mean_auc'], estimator, flattop)
        ax.set(xscale='log', ylim=(0.55, 1.02))
        ax.xaxis.set_minor_locator(NullLocator())

    for ax, scd in ((nc_n, 'non-conjugate'), (c_n, 'conjugate')):
        auc_lines(ax, roc[(roc['Np'] == 64) & (roc['scd'] == scd)], 'N')
    # Only the SSCA and FAM were run at other Np
    for ax, scd in ((nc_np, 'non-conjugate'), (c_np, 'conjugate')):
        auc_lines(ax, roc[(roc['N'] == 4096) & (roc['scd'] == scd) & roc['estimator'].str.contains('ssca|fam')], 'Np')

    titles = [['(a) Cycle leakage vs. $N$', '(b) AUC vs. $N$, non-conj.', '(c) AUC vs. $N$, conjugate'],
              ['(d) Cycle leakage vs. $N_p$', '(e) AUC vs. $N_p$, non-conj.', '(f) AUC vs. $N_p$, conjugate']]
    for row, (axis_name, label) in enumerate((('N', '$N$'), ('Np', '$N_p$'))):
        for column, ax in enumerate(axes[row]):
            if axis_name == 'N':
                power_of_two_axis(ax)
            else:
                plain_log_axis(ax.xaxis, [16, 64, 256])
            ax.set_xlabel(label)
            ax.set_title(titles[row][column], loc='left')
            style(ax)
        axes[row, 1].set_ylabel('Mean AUC')
        axes[row, 2].tick_params(labelleft=False)

    keys = [Line2D([], [], color=color, marker=marker, ms=4.5, mec='white', mew=0.7, label=label)
            for label, color, marker in ESTIMATORS.values() if label != 'S3CA']
    keys += [Line2D([], [], color=MUTED, marker='o', ms=4.5, mec='white', mew=0.7, label='default window'),
             Line2D([], [], color=MUTED, marker='o', ms=4.5, mfc='white', mec=MUTED, mew=0.9, linestyle='--', label='flattop')]
    fig.legend(handles=keys, loc='upper center', ncol=len(keys), frameon=False, handlelength=2.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.84, wspace=0.28, hspace=0.85)
    return fig


def fam_aliasing_figure():
    """Cyclic domain profile of the FAM with flattop a(n) at hop L = Np/4 and Np/8, on the signal of Fig. 4.

    With L = Np/4 the flattop channelizer lets images of each feature through at alpha_0 +- 1/L.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from cdp_figure import cyclic_domain_profile, N, NP, SAMPLES_PER_SYMBOL, CFO
    import scf_test.tests as test
    from scf_test.algorithm import fam_flattop
    from scf_test.psk import create_rect_bpsk_signal

    rng = np.random.default_rng(2026)  # same realization as Fig. 4
    signal = test.add_cfo(create_rect_bpsk_signal(N, SAMPLES_PER_SYMBOL, rng=rng), CFO)
    signal = test.add_awgn_snr(signal, desired_snr=10, rng=rng)[:N]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.6 * CM))
    for k in range(1, 5):
        ax.axvline(k / SAMPLES_PER_SYMBOL, color=GRID, linewidth=0.8, zorder=0)
    for hop, color, label in ((NP // 4, ESTIMATORS['fam'][1], '$L = N_p/4$'), (NP // 8, BLUE_DARK, '$L = N_p/8$')):
        coherence, _, alpha = fam_flattop(signal, Np=NP, L=hop, coherence=True)
        alpha_unique, profile = cyclic_domain_profile(coherence, alpha)
        ax.plot(alpha_unique, profile, color=color, linewidth=0.7, label=label)
    ax.set(xlim=(0, 0.45), ylim=(0, 1.02), xticks=np.arange(0, 0.41, 0.1), xlabel=r'Cycle frequency $\alpha$',
           ylabel='Normalized CDP')
    style(ax)
    ax.grid(False)
    for line in ax.legend(loc='upper right', frameon=False).get_lines():
        line.set_linewidth(1.5)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.21, top=0.97)
    return fig


def cdp_signals_figure():
    """Cycle frequency identification test on the other signal types: cyclic domain profile of one realization.

    Rows: SRRC QPSK, MSK, GMSK. Columns: non-conjugate, conjugate. N = 4096, Np = 64, 10 dB, default windows,
    from the cdp jobs of revision_runs.py. Gray lines mark the features the text refers to.
    """
    CFO, F_SYM = 0.05, 0.1
    marked = {('qpsk', False): [-F_SYM, F_SYM], ('msk', False): [-F_SYM, F_SYM], ('gmsk', False): [-F_SYM, F_SYM],
              ('qpsk', True): [], ('msk', True): [2*CFO + k*F_SYM/2 for k in (-3, -1, 1, 3)],
              ('gmsk', True): [2*CFO + k*F_SYM/2 for k in (-3, -1, 1, 3)]}
    signals = (('qpsk', 'SRRC QPSK'), ('msk', 'MSK'), ('gmsk', 'GMSK'))
    fig, axes = plt.subplots(3, 2, figsize=(TEXT_WIDTH, 9.0 * CM), sharex=True, sharey=True)
    for row, (signal, signal_name) in enumerate(signals):
        for column, conjugate in enumerate((False, True)):
            ax = axes[row, column]
            for cycle_frequency in marked[(signal, conjugate)]:
                ax.axvline(cycle_frequency, color=GRID, linewidth=1.2, zorder=0)
            for estimator in ('fastsc', '2dfft', 'fam', 'ssca'):  # Fast-SC first: its lower floor would hide the rest
                path = DATA / 'cdp' / f"cdp_{estimator}_Np64_N4096_snr10_{'c' if conjugate else 'nc'}_{signal}.npz"
                if not path.exists():  # the Fast-SC has no conjugate SCD
                    continue
                with np.load(path) as data:
                    alpha, profile = data['alpha_0'], data['profile_0']
                order = np.argsort(alpha)
                label, color, _ = ESTIMATORS[estimator]
                ax.plot(alpha[order], profile[order], color=color, linewidth=0.6, alpha=0.85, label=label)
            ax.set(xlim=(-0.5, 0.5), ylim=(0, 1.4))
            style(ax)
            ax.grid(False)
            ax.set_title(f"{signal_name}, {'conjugate' if conjugate else 'non-conjugate'}", loc='left')
    for ax in axes[-1]:
        ax.set_xlabel(r'Cycle frequency $\alpha$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\max_f |C(f, \alpha)|$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(name) for name in ('SSCA', 'FAM', '2D-FFT', 'Fast-SC')]
    legend = fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', ncol=4,
                        frameon=False, bbox_to_anchor=(0.5, 1.0))
    for line in legend.get_lines():
        line.set_linewidth(1.5)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.90, hspace=0.45, wspace=0.06)
    return fig


def cdp_sensitivity_figure():
    """Cycle frequency identification test (Fig. 4) at other N and Np, for the SSCA, FAM, 2D-FFT and Fast-SC.

    Left column: N = 2^10, 2^16 at Np = 64. Right column: Np = 16, 256 at N = 4096 (N = 4096, Np = 64 is Fig. 4).
    One realization of
    the rectangular BPSK of Fig. 4 (10 dB), cyclic domain profile normalized to its value at alpha = 0, alpha >= 0.
    The 2D-FFT is left out at N = 2^16, where it would need O(N^2) memory (about 68 GB).
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from cdp_figure import cyclic_domain_profile, SAMPLES_PER_SYMBOL, CFO
    import scf_test.tests as test
    from scf_test.algorithm import ssca, fam, fast_sc_wrapper, scf_2d_fft_wrapper
    from scf_test.psk import create_rect_bpsk_signal

    settings = [[(2**10, 64), (2**16, 64)], [(4096, 16), (4096, 256)]]
    estimators = {  # drawn in this order: the Fast-SC's lower floor would otherwise hide the rest
        'fastsc': lambda x, Np: fast_sc_wrapper(x, Np=Np, coherence=True),
        '2dfft': lambda x, Np: scf_2d_fft_wrapper(x, Np=Np, coherence=True),
        'fam': lambda x, Np: fam(x, Np=Np, L=Np // 4, coherence=True),
        'ssca': lambda x, Np: ssca(x, Np=Np, coherence=True),
    }
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.0 * CM), sharex=True, sharey=True)
    for column, column_settings in enumerate(settings):
        for row, (N, Np) in enumerate(column_settings):
            rng = np.random.default_rng([2026, N])
            signal = test.add_cfo(create_rect_bpsk_signal(N // SAMPLES_PER_SYMBOL + 1, SAMPLES_PER_SYMBOL, rng=rng), CFO)
            signal = test.add_awgn_snr(signal, desired_snr=10, rng=rng)[:N]
            ax = axes[row, column]
            for k in range(1, 10):
                ax.axvline(k / SAMPLES_PER_SYMBOL, color=GRID, linewidth=1.0, zorder=0)
            for name, estimate in estimators.items():
                if name == '2dfft' and N > 4096:
                    continue
                coherence, _, alpha = estimate(signal, Np)
                alpha_unique, profile = cyclic_domain_profile(coherence, alpha)
                keep = alpha_unique >= 0
                label, color, _ = ESTIMATORS[name]
                ax.plot(alpha_unique[keep], profile[keep], color=color, linewidth=0.6, alpha=0.85, label=label)
            ax.set(xlim=(0, 0.95), ylim=(0, 1.05))
            style(ax)
            ax.grid(False)
            exponent = int(np.log2(N))
            ax.set_title(f'$N = 2^{{{exponent}}}$, $N_p = {Np}$', loc='left')
    for ax in axes[-1]:
        ax.set_xlabel(r'Cycle frequency $\alpha$')
    fig.supylabel('Normalized CDP', fontsize=8, x=0.005)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(name) for name in ('SSCA', 'FAM', '2D-FFT', 'Fast-SC')]
    legend = fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', ncol=4,
                        frameon=False, bbox_to_anchor=(0.5, 1.0))
    for line in legend.get_lines():
        line.set_linewidth(1.5)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.8, hspace=0.55, wspace=0.06)
    return fig


def scalloping_figure():
    """Amplitude of a feature against its distance from the nearest cycle frequency bin (from scalloping.py)."""
    from scalloping import window_response

    table = pd.read_csv(DATA / 'scalloping.csv')
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.6 * CM))
    grid = np.linspace(0, 0.5, 101)
    ax.plot(grid, [window_response('rectangular', d) for d in grid], color='black', label='Rectangular window')
    ax.plot(grid, [window_response('hamming', d) for d in grid], color=AXIS, linestyle='--', label='Hamming window')
    # One colour per configuration; filled markers have a rectangular window along time, hollow ones Hamming
    for name, label, color, marker, nudge, filled in (
            ('Fast-SC', 'Fast-SC', ESTIMATORS['fastsc'][1], 'D', -0.012, True),
            ('SSCA flattop/boxcar', 'SSCA, flattop/boxcar', '#7b4fd6', 'o', 0.0, True),
            ('FAM flattop/boxcar', 'FAM, flattop/boxcar', '#c2185b', 's', 0.012, True),
            ('SSCA', 'SSCA, Hamming', ESTIMATORS['ssca'][1], 'o', -0.006, False),
            ('FAM', 'FAM, Hamming', ESTIMATORS['fam'][1], 's', 0.006, False)):
        stats = table[table['estimator'] == name].groupby('bin_offset')['amplitude'].agg(['mean', 'std'])
        ax.errorbar(stats.index + nudge, stats['mean'], yerr=stats['std'], fmt=marker, color=color, ms=3.5,
                    mfc=color if filled else 'white', mew=0.9, elinewidth=0.6, capsize=1.2, label=label)
    ax.set(xlim=(-0.03, 0.53), ylim=(0.4, 1.1), xlabel='Offset of the feature from the nearest bin (bins)',
           ylabel='Amplitude / theory')
    style(ax)
    ax.legend(loc='lower left', frameon=False, ncol=2, fontsize=6, handletextpad=0.3, columnspacing=0.8)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.21, top=0.97)
    return fig


if __name__ == '__main__':
    # Supplementary Figs. 8-12 (Fig. 13: resolution_figure.py, Fig. 14: robustness_breakdown.py)
    for name, figure in (('supp_sensitivity', sensitivity_figure()), ('supp_convergence', convergence_figure()),
                         ('supp_fam_aliasing', fam_aliasing_figure()), ('supp_scalloping', scalloping_figure()),
                         ('supp_cdp_sensitivity', cdp_sensitivity_figure())):
        figure.savefig(FIG_DIR / f'{name}.pdf')
        figure.savefig(FIG_DIR / f'{name}.png', dpi=300)
        print(f'fig/{name}.pdf and .png written')
