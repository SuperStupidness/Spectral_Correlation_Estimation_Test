"""
scf_test — Spectral Correlation Function benchmark suite.

Quick-start
-----------
    from scf_test import ssca, fam, fast_sc_wrapper, scf_2d_fft_wrapper

All four return ``(Sx, f, alpha)`` with a consistent call signature:

    Sx, f, alpha = ssca(signal, Np=64, conjugate=False)
    Sx, f, alpha = fam(signal, Np=64, L=16, conjugate=False)
    Sx, f, alpha = fast_sc_wrapper(signal, Np=64, conjugate=False)
    Sx, f, alpha = scf_2d_fft_wrapper(signal, Np=64, conjugate=False)

Lower-level entry points (FastSC and 2D-FFT) are also exported for users
who need direct control over algorithm-specific parameters:

    from scf_test import Fast_SC, scf_2d_fft
"""

from .algorithm import (
    ssca,
    fam,
    fast_sc_wrapper,
    scf_2d_fft_wrapper,
    Fast_SC,
    scf_2d_fft,
    psd,
    spectral_correlation_to_coherence,
)
from .tests import compute_cdp, plot_cdp

__all__ = [
    "ssca",
    "fam",
    "fast_sc_wrapper",
    "scf_2d_fft_wrapper",
    "Fast_SC",
    "scf_2d_fft",
    "psd",
    "spectral_correlation_to_coherence",
    "compute_cdp",
    "plot_cdp",
]
