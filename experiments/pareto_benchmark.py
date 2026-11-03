"""Execution times for the Pareto front (Fig. 7), every configuration in one pytest-benchmark session.

    python -m pytest experiments/pareto_benchmark.py --benchmark-only --benchmark-disable-gc \
        --benchmark-warmup=on --benchmark-min-rounds=10 --benchmark-json=data/revision/pareto_benchmark.json

Same signal length (N = 4096) and settings as the robustness test. Run it on an otherwise idle machine.
For the Pareto fronts at other lengths (README), set PARETO_N and write to pareto_benchmark_N<N>.json:

    PARETO_N=16384 python -m pytest experiments/pareto_benchmark.py ... --benchmark-json=data/revision/pareto_benchmark_N16384.json

The 2D-FFT is skipped above N = 4096 (O(N^2) memory) and the Fast-SC above Np = 64 there, as in revision_runs.py.
"""
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scf_test.algorithm import (ssca, ssca_flattop, fam, fam_flattop, fast_sc_wrapper, scf_2d_fft_wrapper,
                                scf_2d_fft_flattop)

N = int(os.environ.get("PARETO_N", 4096))
NPS = [8, 16, 32, 64, 128, 256]
CONFIGURATIONS = {
    "ssca": lambda Np: partial(ssca, Np=Np),
    "ssca_flattop": lambda Np: partial(ssca_flattop, Np=Np),
    "fam": lambda Np: partial(fam, Np=Np, L=Np // 4),
    "fam_flattop": lambda Np: partial(fam_flattop, Np=Np, L=Np // 8),
    "fastsc": lambda Np: partial(fast_sc_wrapper, Np=Np, alpha_max=0.5),
    "2dfft": lambda Np: partial(scf_2d_fft_wrapper, Np=Np),
    "2dfft_flattop": lambda Np: partial(scf_2d_fft_flattop, Np=Np),
}


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, N) + 1j * rng.uniform(-1, 1, N)


@pytest.mark.parametrize("Np", NPS)
@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
def test_time(benchmark, signal, configuration, Np):
    if N > 4096 and (configuration.startswith("2dfft") or (configuration == "fastsc" and Np > 64)):
        pytest.skip("not run at this signal length (see revision_runs.py --pareto-n)")
    benchmark.extra_info.update(configuration=configuration, Np=Np, N=N)
    benchmark(CONFIGURATIONS[configuration](Np), signal)
