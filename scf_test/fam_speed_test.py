import pytest
import numpy as np
from scf_test.algorithm import fam

N = 2**15 #32768

@pytest.mark.parametrize("L", [
    (1),
    (2),
    (4),
    (8),
    (16),
    (32),
    (64),
    (128), # Recommended: Np/4
])

def test_fam(benchmark, L):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(fam, signal, Np=512, L=L, conjugate=False)
