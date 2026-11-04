"""README gallery figures for the resolution tests.

    python experiments/resolution_gallery.py                # both figures
    python experiments/resolution_gallery.py --runs 20      # fewer leakage runs (default 100, ~20-30 min)
    python experiments/resolution_gallery.py --reuse-leakage

fig/ssca_resolution_test.png: resolution_test on the SSCA (half-amplitude dip), the two peaks at the limit.
fig/ssca_window_resolution_test.png: window_resolution_test on ssca_flattop, with the cycle leakage of
extended_window_test for the trade-off panel (its matrix is saved to data/window_leakage_ssca.npy).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scf_test.tests import resolution_test, window_resolution_test, extended_window_test
from scf_test.algorithm import ssca, ssca_flattop


def save_instead_of_show(path):
    """The tests call plt.show(); save the current figure to `path` instead."""
    def show():
        plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"{path.relative_to(ROOT)} written")
    plt.show = show


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=100, help="extended_window_test runs for the leakage")
    parser.add_argument("--reuse-leakage", action="store_true",
                        help="load the leakage matrix saved by an earlier run instead of recomputing it")
    args = parser.parse_args()

    save_instead_of_show(ROOT / "fig" / "ssca_resolution_test.png")
    resolution_test(ssca, name="SSCA", shifts=8)   # finer grid, for a smoother curve

    leakage_file = ROOT / "data" / "window_leakage_ssca.npy"
    if args.reuse_leakage:
        leakage = np.load(leakage_file)
    else:
        leakage_file.parent.mkdir(exist_ok=True)
        leakage = extended_window_test(ssca_flattop, number_of_runs=args.runs, plot=False)
        np.save(leakage_file, leakage)
    save_instead_of_show(ROOT / "fig" / "ssca_window_resolution_test.png")
    window_resolution_test(ssca_flattop, name="SSCA", leakage=leakage)


if __name__ == "__main__":
    main()
