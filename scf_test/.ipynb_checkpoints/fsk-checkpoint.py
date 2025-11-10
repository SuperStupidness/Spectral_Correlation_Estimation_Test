import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from .core.filter_design import gaussian_design
from .visualization.time_domain import plot_fsk_signal

def create_gmsk_signal(number_of_symbols, samples_per_symbol, f_carrier=0.05, BT=0.3, gauss_filter_span=4, plot=False, normalize=False, rng=np.random.default_rng()):
    """
    Generates a GMSK (Gaussian Minimum Shift Keying) modulated signal.

    Parameters:
    -----------
    number_of_symbols : int
        Total number of symbols (bits) to generate.
    samples_per_symbol : int
        Number of samples per symbol duration.
    f_carrier : float, optional
        Carrier frequency normalized to the sampling frequency (default: 0.01).
    BT : float, optional
        Bandwidth-Time product for the Gaussian filter (default: 0.3).
    gauss_filter_span : int, optional
        Span of the Gaussian filter in symbol periods (default: 4).
    plot : bool, optional
        Whether to plot the generated signal (default: False).
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).
        Used to generate the random bit sequence.
    

    Returns:
    --------
    tuple:
        (gmsk_signal, gmsk_symbols, instantaneous_phase)
    """
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")

    n = number_of_symbols * samples_per_symbol
    t = np.linspace(0, n, n)

    modulation_index=0.5

    bit_seq = rng.integers(0, 2, number_of_symbols)
    MSK_symbol = 2*bit_seq - 1
    MSK_symbol_upsampled = np.zeros(n)
    MSK_symbol_upsampled[::samples_per_symbol] = MSK_symbol
    MSK_symbol_gauss = scipy.signal.oaconvolve(MSK_symbol_upsampled, gaussian_design(BT, samples_per_symbol, gauss_filter_span, normalize=normalize), mode="same")
    
    delta_f = 1/(2*samples_per_symbol) * modulation_index

    theta = np.zeros(len(MSK_symbol_gauss))

    theta[1:] = 2*np.pi * np.cumsum(delta_f*MSK_symbol_gauss[:-1]) 

    phi_of_t = 2*np.pi*f_carrier * t + theta

    gmsk_signal = np.exp(1j * phi_of_t)
    gmsk_symbols = MSK_symbol_gauss

    if plot:
        plot_fsk_signal(gmsk_signal, gmsk_symbols, phi_of_t, delta_f, f_carrier, f"GMSK signal with Bandwidth-Time product = {BT}, f_c={f_carrier}")
    
    return gmsk_signal, gmsk_symbols, phi_of_t