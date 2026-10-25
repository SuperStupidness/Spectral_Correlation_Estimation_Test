import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from .core.filter_design import srrc_design, rectangular_design
from .visualization.time_domain import plot_signal
from .visualization.constellation import signal_constellation_generator

def create_rect_apsk_signal(number_of_constellations, number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Creates an APSK modulated signal with rectangular pulse shaping.
    
    Generates either a 16-APSK or 32-APSK signal using random, IID symbols based on
    DVB-S2 standard constellations. The symbols are arranged in concentric rings
    with specific radius ratios for optimized performance. The signal is upsampled
    and filtered with a normalized rectangular pulse shape.
    
    Parameters
    ----------
    number_of_constellations : int
        The order of the APSK constellation (must be either 16 or 32)
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    
    Returns
    -------
    numpy.ndarray
        Complex baseband APSK signal after rectangular pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - 16-APSK uses 2 concentric rings with 4 inner and 12 outer points 
    - 32-APSK uses 3 concentric rings with 4 inner, 12 middle, and 16 outer points 
    - Ring radius ratios follow DVB-S2 specification for code rate 9/10
    - The rectangular pulse shape is normalized to have unit energy
    """
    
    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif number_of_constellations != 16 and number_of_constellations != 32:
        raise ValueError("Error: Number of constellations must be 16 or 32")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    
    # Generate random sequence of symbol
    APSK_map = signal_constellation_generator(str(number_of_constellations) + "APSK")
          
    map_index = rng.integers(0, len(APSK_map), number_of_symbols) 
    APSK_symbols = APSK_map[map_index]
    
    # Upsampling the symbols
    APSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    APSK_symbols_upsampled[::samples_per_symbol] = APSK_symbols
    
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    APSK_signal = scipy.signal.oaconvolve(APSK_symbols_upsampled, rect_pulse_shape, mode="same")

    if plot:
        plot_signal(APSK_signal, f"Rect {number_of_constellations}APSK Signal")
    
    return APSK_signal

def create_srrc_apsk_signal(number_of_constellations, number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an APSK modulated signal with Square Root Raised Cosine (SRRC) pulse shaping.
    
    Generates either a 16-APSK or 32-APSK signal using random, IID symbols based on
    DVB-S2 standard constellations. The symbols are arranged in concentric rings
    with specific radius ratios for optimized performance. The signal is upsampled
    and filtered with an SRRC pulse shape for spectral shaping.
    
    Parameters
    ----------
    number_of_constellations : int
        The order of the APSK constellation (must be either 16 or 32)
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    
    Returns
    -------
    numpy.ndarray
        Complex baseband APSK signal after SRRC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - 16-APSK uses 2 concentric rings with 4 inner and 12 outer points 
    - 32-APSK uses 3 concentric rings with 4 inner, 12 middle, and 16 outer points
    - Ring radius ratios follow DVB-S2 specification for code rate 9/10
    - Requires the srrc_design function to create the pulse shaping filter
    """

    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif number_of_constellations != 16 and number_of_constellations != 32:
        raise ValueError("Error: Number of constellations must be 16 or 32")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    
    # Generate random sequence of symbol
    APSK_map = signal_constellation_generator(str(number_of_constellations) + "APSK")
        
    map_index = rng.integers(0, len(APSK_map), number_of_symbols) 
    APSK_symbols = APSK_map[map_index]
    
    # Upsampling the symbols
    APSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    APSK_symbols_upsampled[::samples_per_symbol] = APSK_symbols
    
    # Apply SRRC pulse shaping
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    APSK_signal = scipy.signal.oaconvolve(APSK_symbols_upsampled, SRRC_pulse_shape, mode="same")

    if plot:
        plot_signal(APSK_signal, f"SRRC {number_of_constellations}APSK Signal")

    return APSK_signal

def create_rc_apsk_signal(number_of_constellations, number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an APSK modulated signal with Square Root Raised Cosine (SRRC) pulse shaping.
    
    Generates either a 16-APSK or 32-APSK signal using random, IID symbols based on
    DVB-S2 standard constellations. The symbols are arranged in concentric rings
    with specific radius ratios for optimized performance. The signal is upsampled
    and filtered with an SRRC pulse shape for spectral shaping.
    
    Parameters
    ----------
    number_of_constellations : int
        The order of the APSK constellation (must be either 16 or 32)
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    
    Returns
    -------
    numpy.ndarray
        Complex baseband APSK signal after SRRC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - 16-APSK uses 2 concentric rings with 4 inner and 12 outer points 
    - 32-APSK uses 3 concentric rings with 4 inner, 12 middle, and 16 outer points
    - Ring radius ratios follow DVB-S2 specification for code rate 9/10
    - Requires the rc_design function to create the pulse shaping filter
    """

    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif number_of_constellations != 16 and number_of_constellations != 32:
        raise ValueError("Error: Number of constellations must be 16 or 32")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    
    # Generate random sequence of symbol
    APSK_map = signal_constellation_generator(str(number_of_constellations) + "APSK")
        
    map_index = rng.integers(0, len(APSK_map), number_of_symbols) 
    APSK_symbols = APSK_map[map_index]
    
    # Upsampling the symbols
    APSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    APSK_symbols_upsampled[::samples_per_symbol] = APSK_symbols
    
    # Apply SRRC pulse shaping
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    APSK_signal = scipy.signal.oaconvolve(APSK_symbols_upsampled, RC_pulse_shape, mode="same")

    if plot:
        plot_signal(APSK_signal, f"RC {number_of_constellations}APSK Signal")

    return APSK_signal