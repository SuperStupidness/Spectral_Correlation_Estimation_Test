import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from .core.filter_design import srrc_design, rectangular_design, rc_design
from .visualization.time_domain import plot_signal
from .visualization.constellation import qam_map_generator

def create_rect_qam_signal(number_of_symbols, samples_per_symbol, number_of_constellations=8, plot=False, rng=np.random.default_rng()):
    """
    Creates a QAM modulated signal with rectangular pulse shaping.
    
    Generates a QAM signal using random, IID symbols selected from a QAM constellation.
    The symbols are upsampled and filtered with a rectangular pulse shape.
    
    Parameters
    ----------
    number_of_constellation : int
        The order of the QAM constellation (e.g., 16 for 16-QAM, 64 for 64-QAM)
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    
    Returns
    -------
    numpy.ndarray
        Complex baseband QAM signal after rectangular pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - Requires the qam_map_generator function to create the QAM constellation
    - Input validation ensures positive, non-zero values for symbols and samples per symbol
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
    QAM_map = qam_map_generator(number_of_constellations)
        
    map_index = rng.integers(0, len(QAM_map), number_of_symbols) 
    QAM_symbols = QAM_map[map_index]
    
    # Upsampling the symbols
    QAM_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    QAM_symbols_upsampled[::samples_per_symbol] = QAM_symbols
    
    # Apply rectangular pulse shaping
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    QAM_signal = scipy.signal.oaconvolve(QAM_symbols_upsampled, rect_pulse_shape, mode="same")

    if plot:
        plot_signal(QAM_signal, f"Rect {number_of_constellations}QAM Signal")

    return QAM_signal

def create_srrc_qam_signal(number_of_symbols, samples_per_symbol, filter_span, beta, number_of_constellations=8, plot=False, rng=np.random.default_rng()):
    """
    Creates a QAM modulated signal with Square Root Raised Cosine (SRRC) pulse shaping.
    
    Generates a QAM signal using random, IID symbols selected from a QAM constellation.
    The symbols are upsampled and filtered with an SRRC pulse shape for spectral shaping.
    
    Parameters
    ----------
    number_of_constellations : int
        The order of the QAM constellation (e.g., 16 for 16-QAM, 64 for 64-QAM)
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
        Complex baseband QAM signal after SRRC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - Requires the qam_map_generator function to create the QAM constellation
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
    QAM_map = qam_map_generator(number_of_constellations)
        
    map_index = rng.integers(0, len(QAM_map), number_of_symbols) 
    QAM_symbols = QAM_map[map_index]
    
    # Upsampling the symbols
    QAM_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    QAM_symbols_upsampled[::samples_per_symbol] = QAM_symbols
    
    # Apply SRRC pulse shaping
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    QAM_signal = scipy.signal.oaconvolve(QAM_symbols_upsampled, SRRC_pulse_shape, mode="same")

    if plot:
        plot_signal(QAM_signal, f"SRRC {number_of_constellations}QAM Signal")

    return QAM_signal

def create_rc_qam_signal(number_of_symbols, samples_per_symbol, filter_span, beta, number_of_constellations=8, plot=False, rng=np.random.default_rng()):
    """
    Creates a QAM modulated signal with Raised Cosine (RC) pulse shaping.
    
    Generates a QAM signal using random, IID symbols selected from a QAM constellation.
    The symbols are upsampled and filtered with an SRRC pulse shape for spectral shaping.
    
    Parameters
    ----------
    number_of_constellations : int
        The order of the QAM constellation (e.g., 16 for 16-QAM, 64 for 64-QAM)
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the RC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    
    Returns
    -------
    numpy.ndarray
        Complex baseband QAM signal after RC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - Requires the qam_map_generator function to create the QAM constellation
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
    QAM_map = qam_map_generator(number_of_constellations)
        
    map_index = rng.integers(0, len(QAM_map), number_of_symbols) 
    QAM_symbols = QAM_map[map_index]
    
    # Upsampling the symbols
    QAM_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    QAM_symbols_upsampled[::samples_per_symbol] = QAM_symbols
    
    # Apply SRRC pulse shaping
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    QAM_signal = scipy.signal.oaconvolve(QAM_symbols_upsampled, RC_pulse_shape, mode="same")

    if plot:
        plot_signal(QAM_signal, f"RC {number_of_constellations}QAM Signal")

    return QAM_signal