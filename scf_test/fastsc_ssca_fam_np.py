import pytest
import numpy as np
from algorithm import ssca, fam, Fast_SC


N = 2**15 #32768

@pytest.mark.parametrize("Np", [
    (8),
    (16),
    (32),
    (64),
    (128),
    (256),
    (512)
])

# The name in test_[algorithm name] matters as run_benchmark function will try to find that name to extract into the dataframe.

# Set the name correctly

def test_ssca(benchmark, Np):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(ssca, signal, Np=Np, conjugate=False)


@pytest.mark.parametrize("Np", [
    (8),
    (16),
    (32),
    (64),
    (128),
    (256),
    (512)
])

def test_fam(benchmark, Np):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(fam, signal, Np=Np, L=Np/4, conjugate=False)

@pytest.mark.parametrize("Np", [
    (8),
    (16),
    (32),
    (64),
    (128),
    (256),
    (512)
])

def test_fastsc(benchmark, Np):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(Fast_SC, signal, Nw=Np, alpha_max=0.5, Fs=1.0, opt = {"abs": 0, "calib": 1, "coh": 0})
