import pytest
from scf_test.algorithm import Fast_SC
import numpy as np

N = 2**15 #32768

@pytest.mark.parametrize("alpha_max", [
    (0.4),
    (0.3),
    (0.2),
    (0.1),
    (0.05),
    (0.01),
    (0.001)
])

def test_fastsc(benchmark, alpha_max):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(Fast_SC, signal, Nw=64, alpha_max=0.5, Fs=1.0, opt = {"abs": 0, "calib": 1, "coh": 0})
