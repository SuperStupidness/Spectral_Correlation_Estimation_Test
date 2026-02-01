import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from .core.filter_design import srrc_design, rectangular_design, rc_design
from .visualization.time_domain import plot_signal
from .visualization.constellation import signal_constellation_generator

def create_rect_bpsk_signal(number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Generate rectangular BPSK signal.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    
    Returns
    -------
    numpy.ndarray
        Complex baseband BPSK signal after rectangular pulse shaping

    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - The rectangular pulse shape is normalized to have unit energy
    """
    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
    
    # Generate random sequence of symbol
    BPSK_map = signal_constellation_generator("BPSK")
        
    map_index = rng.integers(0, len(BPSK_map), number_of_symbols) 
    BPSK_symbols = BPSK_map[map_index]
    
    # Upsampling the symbols
    BPSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    BPSK_symbols_upsampled[::samples_per_symbol] = BPSK_symbols
    
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    BPSK_signal = scipy.signal.oaconvolve(BPSK_symbols_upsampled, rect_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(BPSK_signal, "Rect BPSK Signal")
    
    return BPSK_signal

def create_srrc_qpsk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates a QPSK modulated signal with Square Root Raised Cosine (SRRC) pulse shaping.
    
    Generates a QPSK signal using random, IID symbols selected from a normalized QPSK 
    constellation. The symbols are upsampled and filtered with an SRRC pulse shape
    for spectral shaping.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = 2*filter_span * samples_per_symbol)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband QPSK signal after SRRC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - Requires the srrc_design function to create the pulse shaping filter
    """

    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    
    # Generate random sequence of symbol
    QPSK_map = signal_constellation_generator("QPSK")
        
    map_index = rng.integers(0, len(QPSK_map), number_of_symbols) 
    QPSK_symbols = QPSK_map[map_index]
    
    # Upsampling the symbols
    QPSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    QPSK_symbols_upsampled[::samples_per_symbol] = QPSK_symbols
    
    # Apply SRRC pulse shaping
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    QPSK_signal = scipy.signal.oaconvolve(QPSK_symbols_upsampled, SRRC_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(QPSK_signal, "SRRC QPSK Signal")

    return QPSK_signal

