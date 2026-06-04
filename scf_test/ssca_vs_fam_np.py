import pytest
import numpy as np
from scf_test.algorithm import ssca, fam

N = 2**12 #32768

@pytest.mark.parametrize("Np", [
    (8),
    (16),
    (32),
    (64),
    (128),
    (256),
    (512)
])

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

