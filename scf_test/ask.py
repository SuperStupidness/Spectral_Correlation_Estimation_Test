import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from .core.filter_design import srrc_design, rectangular_design, rc_design
from .visualization.time_domain import plot_signal
from .visualization.constellation import signal_constellation_generator

def create_rect_ask_signal(number_of_constellations, number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Creates an Amplitude Shift Keying (ASK) modulated signal with rectangular pulse shaping.
    
    Parameters:
    -----------
    number_of_constellations : int
        Number of amplitude levels in the modulation scheme (must be a power of 2).
    number_of_symbols : int
        Total number of symbols to generate in the signal.
    samples_per_symbol : int
        Number of samples per symbol duration (upsampling factor).
    plot : bool, optional
        Whether to plot the generated signal (default: False).
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).
    
    Returns:
    --------
    ndarray:
        The complex ASK modulated signal after rectangular pulse shaping.
    
    Notes:
    ------
    The amplitude levels are linearly spaced between -1 and 1.
    Rectangular pulse shaping is applied with energy normalized to 1.
    """
    
    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif np.mod(np.log2(number_of_constellations), 1) != 0 or number_of_constellations <= 0:
        raise ValueError("Error: Number of levels must be a power of 2 for FSK")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    # Generate random sequence of symbol
    ASK_map = signal_constellation_generator(str(number_of_constellations) + "ASK")
        
    map_index = rng.integers(0, len(ASK_map), number_of_symbols) 
    ASK_symbols = ASK_map[map_index]
    
    # Upsampling the symbols
    ASK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    ASK_symbols_upsampled[::samples_per_symbol] = ASK_symbols
    
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    ASK_signal = scipy.signal.oaconvolve(ASK_symbols_upsampled, rect_pulse_shape, mode="same")

    if plot:
        plot_signal(ASK_signal, f"Rect {number_of_constellations}ASK Signal")

    return ASK_signal

def create_srrc_ask_signal(number_of_constellations, number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an Amplitude Shift Keying (ASK) modulated signal with Square Root Raised Cosine pulse shaping.
    
    Parameters:
    -----------
    number_of_constellations : int
        Number of amplitude levels in the modulation scheme (must be a power of 2).
    number_of_symbols : int
        Total number of symbols to generate in the signal.
    samples_per_symbol : int
        Number of samples per symbol duration (upsampling factor).
    filter_span : int
        Span of the SRRC filter in symbol periods.
    beta : float
        Roll-off factor for the SRRC filter (typically between 0 and 1).
    plot : bool, optional
        Whether to plot the generated signal (default: False)..
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).
        
    Returns:
    --------
    ndarray:
        The complex ASK modulated signal after SRRC pulse shaping.

    Notes:
    ------
    The SRRC filter is designed using the srrc_design function.
    The amplitude levels are linearly spaced between -1 and 1.
    """
    
    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif np.mod(np.log2(number_of_constellations), 1) != 0 or number_of_constellations <= 0:
        raise ValueError("Error: Number of levels must be a power of 2 for FSK")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    # Generate random sequence of symbol
    ASK_map = signal_constellation_generator(str(number_of_constellations) + "ASK")
        
    map_index = rng.integers(0, len(ASK_map), number_of_symbols) 
    ASK_symbols = ASK_map[map_index]
    
    # Upsampling the symbols
    ASK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    ASK_symbols_upsampled[::samples_per_symbol] = ASK_symbols
    
    # Create SRRC pulse shaping filter
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    
    ASK_signal = scipy.signal.oaconvolve(ASK_symbols_upsampled, SRRC_pulse_shape, mode="same")

    if plot:
        plot_signal(ASK_signal, f"SRRC {number_of_constellations}ASK Signal")

    return ASK_signal

def create_rc_ask_signal(number_of_constellations, number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an Amplitude Shift Keying (ASK) modulated signal with Raised Cosine pulse shaping.
    
    Parameters:
    -----------
    number_of_constellations : int
        Number of amplitude levels in the modulation scheme (must be a power of 2).
    number_of_symbols : int
        Total number of symbols to generate in the signal.
    samples_per_symbol : int
        Number of samples per symbol duration (upsampling factor).
    filter_span : int
        Span of the RC filter in symbol periods.
    beta : float
        Roll-off factor for the RC filter (typically between 0 and 1).
    plot : bool, optional
        Whether to plot the generated signal (default: False)..
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).
        
    Returns:
    --------
    ndarray:
        The complex ASK modulated signal after RC pulse shaping.

    Notes:
    ------
    The RC filter is designed using the rc_design function.
    The amplitude levels are linearly spaced between -1 and 1.
    """
    
    # Parameters check
    if number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be postive and non zero")
    elif samples_per_symbol <= 0:
        raise ValueError("Error: Samples per symbol must be postive and non zero")
    elif np.mod(np.log2(number_of_constellations), 1) != 0 or number_of_constellations <= 0:
        raise ValueError("Error: Number of levels must be a power of 2 for FSK")
    elif not isinstance(samples_per_symbol, int):
        raise TypeError("Error: Samples per symbol must be an integer")
    elif not isinstance(number_of_symbols, int):
        raise TypeError("Error: Number of symbols must be an integer")
        
    # Generate random sequence of symbol
    ASK_map = signal_constellation_generator(str(number_of_constellations) + "ASK")
        
    map_index = rng.integers(0, len(ASK_map), number_of_symbols) 
    ASK_symbols = ASK_map[map_index]
    
    # Upsampling the symbols
    ASK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    ASK_symbols_upsampled[::samples_per_symbol] = ASK_symbols
    
    # Create SRRC pulse shaping filter
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    
    ASK_signal = scipy.signal.oaconvolve(ASK_symbols_upsampled, RC_pulse_shape, mode="same")

    if plot:
        plot_signal(ASK_signal, f"RC {number_of_constellations}ASK Signal")

    return ASK_signal

def create_rect_ook_signal(number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Creates an On-Off Keying (OOK) modulated signal with rectangular pulse shaping.
    
    Generates an OOK signal using random, IID binary symbols (0 or 1). The symbols 
    are upsampled and filtered with a normalized rectangular pulse shape.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    plot : bool, optional
        Whether to plot the generated signal (default: False)..
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).
    
    Returns
    -------
    numpy.ndarray
        Complex baseband OOK signal after rectangular pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution between 0 and 1
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
    OOK_map = signal_constellation_generator("OOK")
        
    map_index = rng.integers(0, len(OOK_map), number_of_symbols) 
    OOK_symbols = OOK_map[map_index]
    
    # Upsampling the symbols
    OOK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    OOK_symbols_upsampled[::samples_per_symbol] = OOK_symbols
    
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    OOK_signal = scipy.signal.oaconvolve(OOK_symbols_upsampled, rect_pulse_shape, mode="same")

    if plot:
        plot_signal(OOK_signal, f"Rect OOK Signal")
    
    return OOK_signal

def create_srrc_ook_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an On-Off Keying (OOK) modulated signal with SRRC pulse shaping.

    Generates an OOK signal using random, IID binary symbols (0 or 1). The
    symbols are upsampled and then filtered by convolution with a Square-Root
    Raised Cosine (SRRC) pulse shape.

    Parameters
    ----------
    number_of_symbols : int
        The number of OOK symbols (0 or 1) to generate in the signal.
    samples_per_symbol : int
        The number of samples per symbol period (oversampling factor).
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    plot : bool, optional
        Whether to plot the generated signal (default: False)..
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).

    Returns
    -------
    numpy.ndarray
        Complex baseband OOK signal after SRRC pulse shaping.

    Notes
    -----
    - Uses random, IID symbols with a uniform distribution between 0 and 1.
    - Requires the srrcDesign function to create the pulse shaping filter
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
    OOK_map = signal_constellation_generator("OOK")
        
    map_index = rng.integers(0, len(OOK_map), number_of_symbols) 
    OOK_symbols = OOK_map[map_index]
    
    # Upsampling the symbols
    OOK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    OOK_symbols_upsampled[::samples_per_symbol] = OOK_symbols
    
    # Apply SRRC pulse shaping
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    OOK_signal = scipy.signal.oaconvolve(OOK_symbols_upsampled, SRRC_pulse_shape, mode="same")

    if plot:
        plot_signal(OOK_signal, f"SRRC OOK Signal")

    return OOK_signal

def create_rc_ook_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an On-Off Keying (OOK) modulated signal with RC pulse shaping.

    Generates an OOK signal using random, IID binary symbols (0 or 1). The
    symbols are upsampled and then filtered by convolution with a
    Raised Cosine (RC) pulse shape.

    Parameters
    ----------
    number_of_symbols : int
        The number of OOK symbols (0 or 1) to generate in the signal.
    samples_per_symbol : int
        The number of samples per symbol period (oversampling factor).
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    plot : bool, optional
        Whether to plot the generated signal (default: False)..
    rng : numpy.random.Generator, optional
        Random number generator instance (default: np.random.default_rng()).

    Returns
    -------
    numpy.ndarray
        Complex baseband OOK signal after SRRC pulse shaping.

    Notes
    -----
    - Uses random, IID symbols with a uniform distribution between 0 and 1.
    - Requires the rc_design function to create the pulse shaping filter
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
    OOK_map = signal_constellation_generator("OOK")
        
    map_index = rng.integers(0, len(OOK_map), number_of_symbols) 
    OOK_symbols = OOK_map[map_index]
    
    # Upsampling the symbols
    OOK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    OOK_symbols_upsampled[::samples_per_symbol] = OOK_symbols
    
    # Apply SRRC pulse shaping
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    OOK_signal = scipy.signal.oaconvolve(OOK_symbols_upsampled, RC_pulse_shape, mode="same")

    if plot:
        plot_signal(OOK_signal, f"RC OOK Signal")

    return OOK_signal