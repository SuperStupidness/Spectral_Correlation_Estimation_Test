"""
Catalogue of test signals for the cycle frequency identification and robustness tests.

Every signal has a symbol rate of 0.1 (10 samples per symbol) and a carrier offset of 0.05, as in the paper,
and comes with its second-order cycle frequencies from theory:

- Linear modulations, x(t) = sum_k a_k p(t - kT) exp(j 2 pi f_c t):
    non-conjugate  alpha = k f_sym, for all k with a rectangular pulse and only k = 1 with an SRRC pulse
                   (the SRRC spectrum is too narrow for P(f + alpha/2) P*(f - alpha/2) to overlap beyond k = 1)
    conjugate      alpha = 2 f_c + k f_sym (same k), only if E[a_k^2] != 0, i.e. for real constellations
                   (BPSK, ASK, OOK); QPSK, 8PSK, pi/4-DQPSK, QAM and APSK have none
- MSK (CPFSK, h = 0.5):  non-conjugate k f_sym; conjugate 2 f_c + (2m + 1) f_sym / 2
- GMSK (BT = 0.3):       non-conjugate f_sym; conjugate 2 f_c +- f_sym / 2
- Incoherent 2FSK, switching between free-running tones at f_c +- df: the tones add sine-wave components, so
  non-conjugate alpha = k f_sym and +-2 df + k f_sym, conjugate 2 f_c + k f_sym and 2 f_c +- 2 df + k f_sym

The lists keep 0 < alpha < 1 (non-conjugate) and -1 < alpha < 1 (conjugate, where alpha = 0 can be a feature).
roc_alpha_nc and roc_alpha_c are the slices used by the robustness test: a weak but present feature, as in
Table I of the paper, or None where the signal has no feature of that kind.

    from scf_test.signals import SIGNALS, make_signal
    x = make_signal("srrc_16qam", N=4096, snr=0, rng=np.random.default_rng(1))
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .psk import (create_rect_bpsk_signal, create_srrc_bpsk_signal, create_rect_qpsk_signal,
                  create_srrc_qpsk_signal, create_rect_8psk_signal, create_srrc_8psk_signal,
                  create_rect_pi4_dqpsk_signal, create_srrc_pi4_dqpsk_signal)
from .qam import create_rect_qam_signal, create_srrc_qam_signal
from .ask import create_rect_ask_signal, create_srrc_ask_signal, create_rect_ook_signal, create_srrc_ook_signal
from .apsk import create_rect_apsk_signal, create_srrc_apsk_signal
from .fsk import create_cpfsk_signal, create_gmsk_signal, create_incoherent_fsk_signal

SAMPLES_PER_SYMBOL = 10
F_SYM = 1 / SAMPLES_PER_SYMBOL
CFO = 0.05
SPAN = 21          # SRRC and Gaussian filter span in symbols, as in the paper
BETA = 0.35        # SRRC roll-off of the new signals (the paper's SRRC QPSK keeps 0.1)
FSK_DF = 0.02      # tone offset of the incoherent 2FSK


@dataclass(frozen=True)
class TestSignal:
    label: str
    generate: Callable          # (number_of_symbols, rng) -> complex baseband signal, carrier at CFO
    non_conjugate: tuple        # significant non-conjugate cycle frequencies, 0 < alpha < 1
    conjugate: tuple            # significant conjugate cycle frequencies, -1 < alpha < 1
    roc_alpha_nc: Optional[float]
    roc_alpha_c: Optional[float]


def _keep(alphas, conjugate):
    """Cycle frequencies inside the principal range: 0 < alpha < 1, or -1 < alpha < 1 for the conjugate SCD
    (where alpha = 0 is a genuine feature, e.g. 2 f_c - f_sym for a real constellation)."""
    alphas = np.round(np.asarray(alphas, dtype=float), 6) + 0.0
    keep = (np.abs(alphas) < 1 - 1e-9) & ((alphas > 1e-9) | conjugate)
    return tuple(sorted(set(alphas[keep].tolist())))


def _linear(rectangular, real):
    """Cycle frequencies of a linear modulation with a rectangular or SRRC pulse."""
    k = np.arange(-10, 11) if rectangular else np.array([-1, 0, 1])
    non_conjugate = _keep(k * F_SYM, False)
    conjugate = _keep(2 * CFO + k * F_SYM, True) if real else ()
    return non_conjugate, conjugate


def _with_cfo(make):
    return lambda n, rng: make(n, rng) * np.exp(2j * np.pi * CFO * np.arange(n * SAMPLES_PER_SYMBOL))


def _linear_signal(label, make, rectangular, real):
    non_conjugate, conjugate = _linear(rectangular, real)
    roc_nc = 4 * F_SYM if rectangular else F_SYM
    roc_c = (2 * CFO + (4 * F_SYM if rectangular else F_SYM)) if real else None
    return TestSignal(label, _with_cfo(make), non_conjugate, conjugate, round(roc_nc, 6),
                      None if roc_c is None else round(roc_c, 6))


S, B = SAMPLES_PER_SYMBOL, BETA
SIGNALS = {
    # PSK
    "rect_bpsk": _linear_signal("Rect BPSK", lambda n, r: create_rect_bpsk_signal(n, S, rng=r), True, True),
    "srrc_bpsk": _linear_signal("SRRC BPSK", lambda n, r: create_srrc_bpsk_signal(n, S, SPAN, B, rng=r), False, True),
    "rect_qpsk": _linear_signal("Rect QPSK", lambda n, r: create_rect_qpsk_signal(n, S, rng=r), True, False),
    "srrc_qpsk": _linear_signal("SRRC QPSK ($\\beta$ = 0.1)", lambda n, r: create_srrc_qpsk_signal(n, S, SPAN, 0.1, rng=r),
                                False, False),
    "rect_8psk": _linear_signal("Rect 8PSK", lambda n, r: create_rect_8psk_signal(n, S, rng=r), True, False),
    "srrc_8psk": _linear_signal("SRRC 8PSK", lambda n, r: create_srrc_8psk_signal(n, S, SPAN, B, rng=r), False, False),
    "rect_pi4_dqpsk": _linear_signal("Rect $\\pi$/4-DQPSK", lambda n, r: create_rect_pi4_dqpsk_signal(n, S, rng=r),
                                     True, False),
    "srrc_pi4_dqpsk": _linear_signal("SRRC $\\pi$/4-DQPSK",
                                     lambda n, r: create_srrc_pi4_dqpsk_signal(n, S, SPAN, B, rng=r), False, False),
    # QAM
    "rect_16qam": _linear_signal("Rect 16QAM", lambda n, r: create_rect_qam_signal(n, S, 16, rng=r), True, False),
    "srrc_16qam": _linear_signal("SRRC 16QAM", lambda n, r: create_srrc_qam_signal(n, S, SPAN, B, 16, rng=r),
                                 False, False),
    "srrc_64qam": _linear_signal("SRRC 64QAM", lambda n, r: create_srrc_qam_signal(n, S, SPAN, B, 64, rng=r),
                                 False, False),
    # ASK and OOK (real constellations, so they have conjugate features)
    "rect_4ask": _linear_signal("Rect 4ASK", lambda n, r: create_rect_ask_signal(4, n, S, rng=r), True, True),
    "srrc_4ask": _linear_signal("SRRC 4ASK", lambda n, r: create_srrc_ask_signal(4, n, S, SPAN, B, rng=r), False, True),
    "rect_ook": _linear_signal("Rect OOK", lambda n, r: create_rect_ook_signal(n, S, rng=r), True, True),
    "srrc_ook": _linear_signal("SRRC OOK", lambda n, r: create_srrc_ook_signal(n, S, SPAN, B, rng=r), False, True),
    # APSK
    "rect_16apsk": _linear_signal("Rect 16APSK", lambda n, r: create_rect_apsk_signal(16, n, S, rng=r), True, False),
    "srrc_16apsk": _linear_signal("SRRC 16APSK", lambda n, r: create_srrc_apsk_signal(16, n, S, SPAN, B, rng=r),
                                  False, False),
    # Constant-envelope (the generators place the carrier themselves)
    "msk": TestSignal("MSK", lambda n, r: create_cpfsk_signal(n, S, modulation_index=0.5, f_carrier=CFO, rng=r)[0],
                      _keep(np.arange(1, 10) * F_SYM, False),
                      _keep(2 * CFO + (2 * np.arange(-10, 10) + 1) * F_SYM / 2, True),
                      round(F_SYM, 6), round(2 * CFO + 3 * F_SYM / 2, 6)),
    "gmsk": TestSignal("GMSK", lambda n, r: create_gmsk_signal(n, S, f_carrier=CFO, BT=0.3, gauss_filter_span=SPAN,
                                                               rng=r)[0],
                       (round(F_SYM, 6),), _keep([2 * CFO - F_SYM / 2, 2 * CFO + F_SYM / 2], True),
                       round(F_SYM, 6), round(2 * CFO + F_SYM / 2, 6)),
    "fsk2": TestSignal("Incoherent 2FSK", lambda n, r: create_incoherent_fsk_signal(n, S, 2, FSK_DF, CFO, rng=r)[0],
                       _keep(np.concatenate([np.arange(-10, 11) * F_SYM + d for d in (0, 2 * FSK_DF, -2 * FSK_DF)]),
                             False),
                       _keep(np.concatenate([2 * CFO + np.arange(-10, 11) * F_SYM + d
                                             for d in (0, 2 * FSK_DF, -2 * FSK_DF)]), True),
                       round(F_SYM, 6), round(2 * CFO + F_SYM, 6)),
}


def expected_cycle_frequencies(name, conjugate=False):
    """Significant cycle frequencies of a catalogue signal (alpha = 0 excluded)."""
    entry = SIGNALS[name]
    return np.array(entry.conjugate if conjugate else entry.non_conjugate)


def make_signal(name, N, snr=None, rng=None):
    """N samples of a catalogue signal, with AWGN at `snr` dB if given. Returns (noisy, clean)."""
    from .tests import add_awgn_snr
    rng = np.random.default_rng() if rng is None else rng
    clean = SIGNALS[name].generate(N // SAMPLES_PER_SYMBOL + 30, rng)[:N]
    if snr is None:
        return clean, clean
    return add_awgn_snr(clean, desired_snr=snr, rng=rng), clean
