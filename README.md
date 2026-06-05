# Spectral Correlation Estimation Algorithm Tests

Benchmark suite for evaluating cyclostationary Spectral Correlation
Function (SCF) estimators. Tests accuracy, cycle leakage, speed, memory, and
signal detection performance (ROC) against BPSK, QPSK, MSK, and GMSK signals.

Four pure-Python reference implementations are included — **SSCA**, **FAM**,
**FastSC**, and **SCF 2D-FFT** — plus an optional compiled C++ binding for
**S3CA** (see [S3CA](#s3ca-optional) below).

## Package layout

```
scf_test/          Python benchmark package — algorithms + test harness
algorithm/s3ca/    C++ S3CA binding (compiled separately, optional)
```

`scf_test` and `algorithm.s3ca` are two separate packages. All imports below
reflect this split.

## Installation

Requires Python ≥ 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/SuperStupidness/Spectral_Correlation_Estimation_Test.git
cd Spectral_Correlation_Estimation_Test
uv sync
```

This installs the pure-Python benchmark suite. S3CA requires a separate
compilation step — see [S3CA](#s3ca-optional).

## Testing Your Own Algorithm

### 1. Required API

Your algorithm must return `(Sx, f, alpha)` — the SCF estimate and its
coordinate grids:

```python
Sx, f, alpha = my_algorithm(signal, Np=64, conjugate=False)
```

| Argument | Type | Description |
|---|---|---|
| `signal` | 1-D complex ndarray | Input IQ samples |
| `Np` | int | Window / channel count (your algorithm's resolution parameter) |
| `conjugate` | bool | If `True`, estimate the conjugate SCF |

| Return | Shape | Description |
|---|---|---|
| `Sx` | (Np, N) or 1-D sparse | SCF magnitudes |
| `f` | same as `Sx` | Normalised spectral frequency, in [-0.5, 0.5) |
| `alpha` | same as `Sx` | Cyclic frequency, in [-1, 1) |

`f` and `alpha` are coordinate grids — every element of `Sx` has a
corresponding `f` and `alpha` value. The shape does not need to be
rectangular; sparse 1-D outputs work fine.

**Spectral coherence** — The ROC test calls your algorithm with
`coherence=True` and expects the normalised SCF in return:

```python
Sx, f, alpha = my_algorithm(signal, Np=64, conjugate=False, coherence=True)
```

If your algorithm does not support coherence, skip the ROC tests with
`skip="roc"`.

**Hop / decimation parameter** — If your algorithm has an `L` parameter
(like FAM), pass `fam=True` to `run_all_tests`. The harness will then call
`my_algorithm(signal, Np=Np, L=L, ...)`.

### 2. Run All Tests

```python
from scf_test.tests import run_all_tests

run_all_tests(my_algorithm, name="my_algo", Np=64)
```

`run_all_tests` returns a dict keyed by test name. Full signature:

```python
run_all_tests(
    func_lambda,          # your algorithm
    name="algorithm",     # label used in plots and saved files
    Np=64,                # window size passed to your algorithm
    L=16,                 # hop size (only used when fam=True)
    N_roc=4096,           # signal length for ROC test
    mode="full",          # "full" or "limited" (see below)
    save=False,           # save plots to fig/
    fam=False,            # pass L= to your algorithm
    skip=None,            # skip tests by name (str or list of str)
)
```

**`mode="full"`** — tests both conjugate and non-conjugate SCF over the full
cycle frequency range `|α| ≤ 1`.

**`mode="limited"`** — non-conjugate only, `|α| ≤ 0.5`. Use this for
algorithms that only output half the cycle frequency range (e.g. FastSC).

**`skip`** — case-insensitive substring match on the test name. Pass a string
or list of strings. For example, the SCF 2D-FFT algorithm works fine at
`N_roc=4096` but its O(N²) memory requirement makes the validation sweep (up to N=2¹⁹)
infeasible unless you have sufficient RAM:

```python
run_all_tests(my_algorithm, name="my_algo", skip="validation")
run_all_tests(my_algorithm, name="my_algo", skip=["validation", "memory"])
```

### 3. What Each Test Measures

| Test | What it checks |
|---|---|
| **Validation** | RMSE against theoretical BPSK SCF across signal lengths $2^10$ – $2^19$ |
| **Memory** | Peak RAM usage vs signal length via `tracemalloc` |
| **Speed** | Mean execution time via `timeit` (10 runs) |
| **Cycle Leakage** | Average SCF magnitude at non-cyclic frequencies — lower is better |
| **ROC (non-conjugate)** | Pd vs Pfa for Rect BPSK, SRRC QPSK, and GMSK at SNR = 0, −5, −10 dB. Requires `coherence=True`. |
| **ROC (conjugate)** | Pd vs Pfa for Rect BPSK, MSK, and GMSK at SNR = 0, −5, −10 dB. Requires `coherence=True`. |

### 4. Running Tests Individually

Each test function is importable from `scf_test.tests`:

```python
from scf_test.tests import (
    validation_test,
    memory_test,
    window_test,
    plot_roc_non_conjugate,
    plot_roc_conjugate,
    run_benchmark_timeit,
)
```

See the example notebooks for full usage with plots.

### 5. Plotting the Cyclic Domain Profile

A quick sanity check before running the full suite:

```python
from scf_test.tests import compute_cdp, plot_cdp

Sx, f, alpha = my_algorithm(signal, Np=64)
plot_cdp({'my_algo': (Sx, alpha)}, normalize=True, alpha_range=(-0.5, 0.5))
```

Overlay multiple algorithms on one axis by passing a dict:

```python
plot_cdp(
    {'my_algo': (Sx_mine, alpha_mine), 'SSCA': (Sx_ssca, alpha_ssca)},
    normalize=True,
)
```

## Example Notebooks

| Notebook | Contents |
|---|---|
| `run_all_test_example.ipynb` | One-call `run_all_tests` for any algorithm |
| `validation_test_example.ipynb` | Correctness check and coherence check |
| `cycle_leakage_test_example.ipynb` | Cycle leakage and window selection |
| `speed_memory_test.ipynb` | pytest-benchmark speed + memory sweeps |
| `roc_test_example.ipynb` | ROC curves across signal types and SNR |

## Reference Algorithms

```python
from scf_test import ssca, fam, fast_sc_wrapper, scf_2d_fft_wrapper

Sx, f, alpha = ssca(signal, Np=64, conjugate=False)
Sx, f, alpha = fam(signal, Np=64, L=16, conjugate=False)
Sx, f, alpha = fast_sc_wrapper(signal, Np=64, conjugate=False)
Sx, f, alpha = scf_2d_fft_wrapper(signal, Np=64, conjugate=False)
```

## S3CA (Optional)

S3CA is a sparse C++ implementation. Prebuilt Windows x86_64 DLLs are
included in `algorithm/s3ca/_prebuilt/` — no compilation needed on Windows.

```python
from algorithm.s3ca import s3ca
Sx, f, alpha = s3ca(signal, Np=64)                   # Np must be 64
Sx, f, alpha = s3ca(signal, Np=64, coherence=True)   # spectral coherence
```

S3CA outputs a sparse 1-D result. `Np` must be 64 — the C++ source hardcodes
a 64-element Chebyshev window in `SSCA.h`. `conjugate` is not yet supported.

**Recompiling from source** (MSYS2 UCRT64 shell — only needed if you are not
on Windows x86_64, or want to rebuild):

```bash
cd algorithm/s3ca/src
g++ -O3 -shared python_bridge.cpp autossca.cpp computefourier.cc \
    filters.cc parameters.cc utils.cc fftw.cc \
    -o s3ca.dll -lfftw3f
```

Do **not** include `run_autossca.cpp` — it has its own `main()` and will
cause linker errors.

Copy `s3ca.dll` and the four MSYS2 runtime DLLs
(`libwinpthread-1.dll`, `libgcc_s_seh-1.dll`, `libstdc++-6.dll`,
`libfftw3f-3.dll`) from `ucrt64/bin/` into `algorithm/s3ca/_prebuilt/`.

## Gallery

*Validation*

![SSCA Validation](/fig/ssca_validation_1024.png)
![SSCA Validation](/fig/ssca_validation_8192.png)
![SSCA Validation](/fig/ssca_validation_65536.png)
![SSCA Validation](/fig/ssca_validation_524288.png)

*Cycle Leakage Test*

![SSCA Cycle Leakage](/fig/ssca_cycle_leakage_test.png)

*Speed and Memory*

![Speed Test](/fig/all_algo_benchmark.png)
![Memory Test](/fig/memory_test.png)

*Performance Test*

![Performance Test](/fig/ssca_roc_test.png)

*Coherence Validation*

![SSCA Coherence Validation](/fig/ssca_coh_validation_1024.png)
![SSCA Coherence Validation](/fig/ssca_coh_validation_8192.png)
![SSCA Coherence Validation](/fig/ssca_coh_validation_65536.png)
![SSCA Coherence Validation](/fig/ssca_coh_validation_262144.png)

*Cyclic Domain Profile Test*

![SSCA CDP](/fig/cdp_test.png)

*Window Test (Extended Cycle Leakage Test)*

![SSCA Window Test](/fig/ssca_window_test.png)

*Extended Validation Test*

![SSCA Extended Validation](/fig/ssca_extended_validation_test.png)

## References

- R. S. Roberts, W. A. Brown, H. H. Loomis, "Computationally efficient algorithms for cyclic spectral analysis," *IEEE Signal Process. Mag.*, vol. 8, no. 2, pp. 38–49, 1991. (SSCA, FAM)
- J. Antoni, G. Xin, N. Hamzaoui, "Fast computation of the spectral correlation," *Mech. Syst. Signal Process.*, vol. 92, pp. 248–277, 2017. (FastSC)
- R. E. B. A. Barros, [RBFastSC](https://github.com/rodrigoel/RBFastSC) — Python implementation of FastSC.
- T. Shevgunov, E. Efimov, "Two-dimensional FFT Algorithm for Estimating Spectral Correlation Function of Cyclostationary Random Processes," *SYNCHROINFO*, 2019. (SCF 2D-FFT)
- Carol Jingyi Li et al., "S3CA: Sparse Strip Spectral Correlation Analyzer," *IEEE Signal Process. Lett.*, 2024. (S3CA)

---

*This README was written by [Claude](https://claude.ai) (Anthropic).*
