"""Execution times for the Pareto front (Fig. 7), every configuration in one pytest-benchmark session.

    python -m pytest experiments/pareto_benchmark.py --benchmark-only --benchmark-disable-gc \
        --benchmark-warmup=on --benchmark-min-rounds=10 --benchmark-json=data/revision/pareto_benchmark.json

Same signal length (N = 4096) and settings as the robustness test. Run it on an otherwise idle machine.
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scf_test.algorithm import (ssca, ssca_flattop, fam, fam_flattop, fast_sc_wrapper, scf_2d_fft_wrapper,
                                scf_2d_fft_flattop)

N = 4096
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
    benchmark.extra_info.update(configuration=configuration, Np=Np)
    benchmark(CONFIGURATIONS[configuration](Np), signal)
