"""S3CA: Sparse Strip Spectral Correlation Analyzer.

Loads the prebuilt s3ca.dll via ctypes.
The DLL and its MSYS2/UCRT64 runtime dependencies must all sit in _prebuilt/.

Build instructions (MSYS2 UCRT64 shell)
----------------------------------------
From algorithm/s3ca/src/, with g++ on PATH:

    g++ -O3 -shared python_bridge.cpp autossca.cpp computefourier.cc \\
        filters.cc parameters.cc utils.cc fftw.cc \\
        -o s3ca.dll -lfftw3f

Copy the resulting s3ca.dll plus these three runtime DLLs from
MSYS2's ucrt64/bin/ (or mingw64/bin/) into algorithm/s3ca/_prebuilt/:

    libwinpthread-1.dll
    libgcc_s_seh-1.dll
    libstdc++-6.dll
    libfftw3f-3.dll

Notes
-----
- Do NOT include run_autossca.cpp — it has its own main() and will cause
  linker errors.
- Arg 6 of compute_s3ca_in_memory must be int seeds (c_int), not a float.
  A float silently truncates the RNG seed to 0 and produces wrong
  magnitudes while still finding the right peak positions.
- Np must be 64. SSCA.h hardcodes a Chebyshev window array of length 64;
  other values will produce garbage or crash.
"""
from __future__ import annotations
import ctypes
import warnings
from pathlib import Path

import numpy as np
import scipy.signal

__all__ = ["s3ca"]

_HERE = Path(__file__).resolve().parent
_DLL_DIR = _HERE / "_prebuilt"
_lib = None


def _load():
    """Load s3ca.dll and its MinGW runtime dependencies, once."""
    global _lib
    if _lib is not None:
        return _lib

    # Pre-load runtime deps bottom-up so the OS loader picks up the MSYS2
    # versions before any incompatible copies on PATH (e.g. Anaconda's
    # libstdc++).
    for dep in ["libwinpthread-1.dll", "libgcc_s_seh-1.dll",
                "libstdc++-6.dll", "libfftw3f-3.dll"]:
        path = _DLL_DIR / dep
        if path.exists():
            ctypes.CDLL(str(path))

    dll_path = _DLL_DIR / "s3ca.dll"
    if not dll_path.exists():
        raise ImportError(
            f"s3ca.dll not found at {dll_path}.\n"
            "Build it from algorithm/s3ca/src/ using:\n\n"
            "    g++ -O3 -shared python_bridge.cpp autossca.cpp "
            "computefourier.cc \\\n"
            "        filters.cc parameters.cc utils.cc fftw.cc \\\n"
            "        -o s3ca.dll -lfftw3f\n\n"
            "Then copy s3ca.dll and the MSYS2 runtime DLLs into "
            "algorithm/s3ca/_prebuilt/."
        )

    _lib = ctypes.CDLL(str(dll_path))
    _lib.compute_s3ca_in_memory.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,    # data_length
        ctypes.c_int,    # size_of_n
        ctypes.c_int,    # np_channels
        ctypes.c_int,    # seeds  (must be c_int, NOT c_float)
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,    # max_out_size
    ]
    _lib.compute_s3ca_in_memory.restype = ctypes.c_int
    return _lib


def s3ca(signal, Np=64, conjugate=False, coherence=False, seed=23,
         max_out_size=None):
    """Sparse SCD estimate via S3CA. Returns (Sx, f, alpha), all 1-D.

    Parameters
    ----------
    signal : array_like, complex
        Input IQ samples. Length must be >= Np. If not a power of two,
        it is truncated to the largest power of two that fits.
    Np : int
        Number of frequency channels. Must be 64 (hardcoded in C++).
    conjugate : bool
        Not yet implemented; placeholder for test-bench compatibility.
    coherence : bool
        Not yet implemented; placeholder for test-bench compatibility.
    seed : int, default 23
        RNG seed passed to the sparse recovery step.
    max_out_size : int, optional
        Maximum number of output coefficients. Defaults to Np * size_of_n.

    Returns
    -------
    Sx : (K,) float32 ndarray
        Sparse coefficient magnitudes (phase is discarded in C++ by design).
    f : (K,) float32 ndarray
        Normalised spectral frequency, f = idx_freq/Np - 0.5, in [-0.5, 0.5).
    alpha : (K,) float32 ndarray
        Cyclic frequency, alpha = idx_alpha/size_of_n - 1.0, in [-1, 1).
    """
    if Np != 64:
        raise NotImplementedError(
            "S3CA hardcodes Np=64 in a Chebyshev window array in SSCA.h. "
            "Other values are not supported."
        )

    if conjugate:
        raise NotImplementedError(
            "conjugate is not yet implemented in the C++ bridge."
        )

    lib = _load()

    sig = np.ascontiguousarray(signal)
    if sig.ndim != 1:
        raise ValueError("signal must be 1-D")

    n_in = len(sig)
    if n_in < Np:
        raise ValueError(f"signal length {n_in} must be >= Np = {Np}")

    # Truncate to the largest power of two (matches floor_to_pow2 in C++).
    size_of_n = 1 << (n_in.bit_length() - 1)
    if size_of_n != n_in:
        warnings.warn(
            f"Signal length {n_in} is not a power of two; "
            f"truncating to {size_of_n} samples.",
            stacklevel=2,
        )
        sig = sig[:size_of_n]

    i_data = np.ascontiguousarray(sig.real, dtype=np.float32)
    q_data = np.ascontiguousarray(sig.imag, dtype=np.float32)

    if max_out_size is None:
        max_out_size = Np * size_of_n
    out_keys = np.zeros(max_out_size, dtype=np.uint32)
    out_values = np.zeros(max_out_size, dtype=np.float32)

    n_found = lib.compute_s3ca_in_memory(
        i_data, q_data, size_of_n,
        size_of_n, int(Np), int(seed),
        out_keys, out_values, int(max_out_size),
    )

    if n_found == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty

    keys = out_keys[:n_found].astype(np.int64)
    Sx = out_values[:n_found].copy()
    idx_alpha, idx_freq = np.divmod(keys, Np)
    f = idx_freq.astype(np.float32) / Np - 0.5
    alpha = idx_alpha.astype(np.float32) / size_of_n - 1.0

    if coherence:
        fs = 1
        psd_tsm = np.fft.ifftshift(psd(signal, L=Np, method="welch", db=False, plot=False))
        sample_1 = np.rint((f + alpha / 2) * Np / fs).astype(int)
        sample_2 = np.rint((f - alpha / 2) * Np / fs).astype(int)
        coherence_denominator = np.sqrt(psd_tsm[sample_1] * psd_tsm[sample_2])
        Sx = Sx / coherence_denominator

    return Sx, f, alpha


def psd(signal, fs=1, method='welch', L=256, db=False, plot=False):
    """Power spectral density estimate. Mirrors scf_test.algorithm.psd."""
    N = len(signal)

    match method.lower():
        case 'daniell':
            if L > N:
                L = int(N * 0.02)
            power_spectrum = 1 / (N * fs) * np.abs(np.fft.fftshift(np.fft.fft(signal))) ** 2
            if L % 2 == 0:
                L = L + 1
            g = np.ones(L) / L
            psd_arr = scipy.signal.oaconvolve(power_spectrum, g)
            start = int((L - 1) / 2)
            psd_truncated = np.zeros(N)
            psd_truncated[:start] = psd_arr[L - 1]
            psd_truncated[start:-start] = psd_arr[L - 1:-(L - 1)]
            psd_truncated[-start:] = psd_arr[-(L - 1)]
            psd_arr = psd_truncated

        case 'bartlett':
            power_sum = np.zeros(L, dtype="float")
            num_time_segments = int(np.ceil(N / L))
            for i in range(num_time_segments):
                power_spectrum = 1 / (L * fs) * np.abs(
                    np.fft.fftshift(np.fft.fft(signal[i * L:min((i + 1) * L, N)], n=L))
                ) ** 2
                power_sum += power_spectrum
            psd_arr = power_sum / num_time_segments

        case 'welch':
            w = np.hanning(L)
            u = 1 / L * np.sum(w ** 2)
            power_sum = np.zeros(L, dtype="float")
            hop = L // 2 if L % 2 == 0 else (L - 1) // 2
            num_time_segments = N // hop
            signal_padded = np.pad(signal, (0, L - (N - num_time_segments * hop)))
            for i in range(num_time_segments):
                power_spectrum = 1 / (L * fs * u) * np.abs(
                    np.fft.fftshift(np.fft.fft(signal_padded[i * hop:(i + 2) * hop] * w))
                ) ** 2
                power_sum += power_spectrum
            psd_arr = power_sum / num_time_segments

        case _:
            raise ValueError(f"Unknown PSD method {method!r}. Use 'daniell', 'bartlett', or 'welch'.")

    if db:
        psd_arr = 10 * np.log10(psd_arr + 1e-10)

    return psd_arr
