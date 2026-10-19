"""Validation figure of the paper (RMSE against signal length), from the results of revision_runs.py.

    python experiments/revision_summary.py     # writes the table this script reads
    python experiments/validation_figure.py

Writes validation_test_rev (magnitudes compared, as in the paper) and
validation_test_rev_complex (complex values compared) as .png and .pdf to fig/.
Same layout as the original figure drawn in Examples.ipynb.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'data' / 'revision' / 'summary' / 'validation_rmse.csv'
FIG_DIR = ROOT / 'fig'
NP = 512

# Label and line style per estimator, as in the original figure
ESTIMATORS = {'ssca': ('SSCA', 'x--'), 'fam': ('FAM', 'o--'), 'fastsc': ('FastSC', '^--')}


def validation_figure(table, column):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    for ax, scd, title in zip(axes, ('non-conjugate', 'conjugate'), ('Non-conjugate', 'Conjugate')):
        for name, (label, line) in ESTIMATORS.items():
            rows = table[(table['estimator'] == name) & (table['scd'] == scd)].sort_values('signal_length')
            if not rows.empty:  # the Fast-SC has no conjugate SCD
                ax.plot(rows['signal_length'], rows[column], line, label=label)
        ax.set_xscale('log', base=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Signal Length (log2 scale)")
        ax.set_ylabel("RMSE")
        ax.set_title(title, fontsize=10)
        ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    table = pd.read_csv(TABLE)
    table = table[table['Np'] == NP]
    print(table.groupby(['estimator', 'scd'])['trials'].first().to_string())
    for name, column in (('validation_test_rev', 'rmse_mean'), ('validation_test_rev_complex', 'rmse_complex_mean')):
        figure = validation_figure(table, column)
        figure.savefig(FIG_DIR / f'{name}.png', dpi=300)
        figure.savefig(FIG_DIR / f'{name}.pdf')
        print(f'fig/{name}.png and .pdf written')
