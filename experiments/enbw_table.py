"""Cycle leakage against the equivalent noise bandwidth (ENBW) of g(n), for the supplementary table.

    python experiments/enbw_table.py

SSCA, flattop a(n), N = 4096, Np = 64, non-conjugate. For each g(n): its ENBW (in bins of 1/N), the square root
of that, and the median cycle leakage relative to a boxcar g(n), on the rectangular BPSK at 0 dB (from the window
search, data/revision/windows, 100 runs) and on noise alone at the same noise power (computed here, 100 runs).
Prints the rows of the LaTeX table.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import scipy.signal

from revision_runs import WINDOWS, leakage_region
from scf_test.algorithm import ssca_flattop

N, NP, RUNS = 4096, 64, 100
SHOWN = ["boxcar", ("tukey", 0.2), "hamming", "hann", "blackmanharris", ("kaiser", 14), "flattop"]
LABELS = {"boxcar": "Boxcar", "('tukey', 0.2)": "Tukey (0.2)", "hamming": "Hamming", "hann": "Hann",
          "blackmanharris": "Blackman--Harris", "('kaiser', 14)": "Kaiser (14)", "flattop": "Flattop"}


def enbw(window):
    w = scipy.signal.get_window(window, N)
    return N * np.sum(w**2) / np.sum(w)**2


def noise_only(window, rng):
    """Median cycle leakage of the SSCA with flattop a(n) and this g(n) on unit-power complex white noise."""
    region, values = None, []
    for _ in range(RUNS):
        x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
        scd, _, alpha = ssca_flattop(x, Np=NP, window_1="flattop", window_2=window)
        if region is None:
            region = leakage_region(alpha, N, False)
        values.append(np.mean(np.abs(scd).ravel()[region]))
    return np.median(values)


if __name__ == "__main__":
    stored = np.load(ROOT / "data" / "revision" / "windows" / "windows_ssca_Np64_N4096_snr0_nc_bpsk.npz")
    names = [str(w) for w in stored["windows"]]
    bpsk = np.median(stored["per_run"][names.index("flattop")], axis=-1)  # leakage for each g(n), a(n) = flattop
    noise = {str(w): noise_only(w, np.random.default_rng(2026)) for w in SHOWN}  # same noise for every g(n)

    for w in SHOWN:
        key = str(w)
        print(f"{LABELS[key]} & {enbw(w):.2f} & {np.sqrt(enbw(w)):.2f} & "
              f"{bpsk[names.index(key)] / bpsk[names.index('boxcar')]:.2f} & {noise[key] / noise['boxcar']:.2f} \\\\")
    ratio = [bpsk[j] / bpsk[names.index('boxcar')] / np.sqrt(enbw(WINDOWS[j])) for j in range(len(WINDOWS))]
    print(f"% all 17 g(n), BPSK: measured / sqrt(ENBW) from {min(ratio):.3f} to {max(ratio):.3f}")
