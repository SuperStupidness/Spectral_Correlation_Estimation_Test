import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from ..core.filter_design import srrc_design, rectangular_design, rc_design
from ..visualization.time_domain import plot_signal
from ..visualization.constellation import signal_constellation_generator

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

def create_srrc_bpsk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Generate SRRC (square-root raised cosine) BPSK signal.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband BPSK signal after SRRC pulse shaping

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
    BPSK_map = signal_constellation_generator("BPSK")
        
    map_index = rng.integers(0, len(BPSK_map), number_of_symbols) 
    BPSK_symbols = BPSK_map[map_index]
    
    # Upsampling the symbols
    BPSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    BPSK_symbols_upsampled[::samples_per_symbol] = BPSK_symbols
    
    # Apply SRRC pulse shaping
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    BPSK_signal = scipy.signal.oaconvolve(BPSK_symbols_upsampled, SRRC_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(BPSK_signal, "SRRC BPSK Signal")

    return BPSK_signal

def create_rc_bpsk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Generate RC (raised cosine) BPSK signal.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the RC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the RC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband BPSK signal after RC pulse shaping

    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
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
    BPSK_map = signal_constellation_generator("BPSK")
        
    map_index = rng.integers(0, len(BPSK_map), number_of_symbols) 
    BPSK_symbols = BPSK_map[map_index]
    
    # Upsampling the symbols
    BPSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    BPSK_symbols_upsampled[::samples_per_symbol] = BPSK_symbols
    
    # Apply SRRC pulse shaping
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    BPSK_signal = scipy.signal.oaconvolve(BPSK_symbols_upsampled, RC_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(BPSK_signal, "RC BPSK Signal")

    return BPSK_signal

def create_rect_qpsk_signal(number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Creates a QPSK modulated signal with rectangular pulse shaping.
    
    Generates a QPSK signal using random, IID symbols selected from a normalized QPSK 
    constellation. The symbols are upsampled and filtered with a normalized 
    rectangular pulse shape.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband QPSK signal after rectangular pulse shaping
    
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
    QPSK_map = signal_constellation_generator("QPSK")
        
    map_index = rng.integers(0, len(QPSK_map), number_of_symbols) 
    QPSK_symbols = QPSK_map[map_index]
    
    # Upsampling the symbols
    QPSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    QPSK_symbols_upsampled[::samples_per_symbol] = QPSK_symbols
    
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    QPSK_signal = scipy.signal.oaconvolve(QPSK_symbols_upsampled, rect_pulse_shape, "same")

    # Plot the signal if requested
    if plot:
        plot_signal(QPSK_signal, "Rect QPSK Signal")
    
    return QPSK_signal

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

def create_rc_qpsk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates a QPSK modulated signal with Raised Cosine (RC) pulse shaping.
    
    Generates a QPSK signal using random, IID symbols selected from a normalized QPSK 
    constellation. The symbols are upsampled and filtered with an RC pulse shape
    for spectral shaping.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the RC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the RC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband QPSK signal after RC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
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
    QPSK_map = signal_constellation_generator("QPSK")
        
    map_index = rng.integers(0, len(QPSK_map), number_of_symbols) 
    QPSK_symbols = QPSK_map[map_index]
    
    # Upsampling the symbols
    QPSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    QPSK_symbols_upsampled[::samples_per_symbol] = QPSK_symbols
    
    # Apply SRRC pulse shaping
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    QPSK_signal = scipy.signal.oaconvolve(QPSK_symbols_upsampled, RC_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(QPSK_signal, "RC QPSK Signal")

    return QPSK_signal

def create_rect_8psk_signal(number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Creates an 8-PSK modulated signal with rectangular pulse shaping.
    
    Generates an 8-PSK signal using random, IID symbols selected from a unit-energy 
    8-PSK constellation. The symbols are upsampled and filtered with a normalized 
    rectangular pulse shape.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband 8-PSK signal after rectangular pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - 8-PSK constellation points are arranged in a circle with unit radius at equal angular spacing
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
    PSK_map = signal_constellation_generator("8PSK")
        
    map_index = rng.integers(0, len(PSK_map), number_of_symbols) 
    PSK_symbols = PSK_map[map_index]
    
    # Upsampling the symbols
    PSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    PSK_symbols_upsampled[::samples_per_symbol] = PSK_symbols
    
    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    PSK_signal = scipy.signal.oaconvolve(PSK_symbols_upsampled, rect_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(PSK_signal, "Rect 8PSK Signal")
    
    return PSK_signal

def create_srrc_8psk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an 8-PSK modulated signal with Square Root Raised Cosine (SRRC) pulse shaping.
    
    Generates an 8-PSK signal using random, IID symbols selected from a unit-energy 
    8-PSK constellation. The symbols are upsampled and filtered with an SRRC pulse shape
    for spectral shaping.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband 8-PSK signal after SRRC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - 8-PSK constellation points are arranged in a circle with unit radius at equal angular spacing
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
    PSK_map = signal_constellation_generator("8PSK")
        
    map_index = rng.integers(0, len(PSK_map), number_of_symbols) 
    PSK_symbols = PSK_map[map_index]
    
    # Upsampling the symbols
    PSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    PSK_symbols_upsampled[::samples_per_symbol] = PSK_symbols
    
    # Apply SRRC pulse shaping
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    PSK_signal = scipy.signal.oaconvolve(PSK_symbols_upsampled, SRRC_pulse_shape, mode="same")
    
    # Plot the signal if requested
    if plot:
        plot_signal(PSK_signal, "SRRC 8PSK Signal")
       
    return PSK_signal

def create_rc_8psk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates an 8-PSK modulated signal with Raised Cosine (RC) pulse shaping.
    
    Generates an 8-PSK signal using random, IID symbols selected from a unit-energy 
    8-PSK constellation. The symbols are upsampled and filtered with an RC pulse shape
    for spectral shaping.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the RC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the RC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband 8-PSK signal after RC pulse shaping
    
    Notes
    -----
    - Uses random, IID symbols with uniform distribution across all constellation points
    - 8-PSK constellation points are arranged in a circle with unit radius at equal angular spacing
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
    PSK_map = signal_constellation_generator("8PSK")
        
    map_index = rng.integers(0, len(PSK_map), number_of_symbols) 
    PSK_symbols = PSK_map[map_index]
    
    # Upsampling the symbols
    PSK_symbols_upsampled = np.zeros(number_of_symbols*samples_per_symbol,dtype=complex)
    PSK_symbols_upsampled[::samples_per_symbol] = PSK_symbols
    
    # Apply SRRC pulse shaping
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    PSK_signal = scipy.signal.oaconvolve(PSK_symbols_upsampled, RC_pulse_shape, mode="same")
    
    # Plot the signal if requested
    if plot:
        plot_signal(PSK_signal, "RC 8PSK Signal")
       
    return PSK_signal

def create_rect_pi4_dqpsk_signal(number_of_symbols, samples_per_symbol, plot=False, rng=np.random.default_rng()):
    """
    Creates a π/4-DQPSK modulated signal with Rectangular pulse shaping.
    
    Generates a differentially encoded π/4-DQPSK signal using random, IID symbols. 
    The modulation alternates between two QPSK constellations offset by π/4 radians,
    with phase transitions determined by one of four possible phase differences.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband π/4-DQPSK signal after Rectangular pulse shaping
        
    π/4-DQPSK implementation based on assistance from Claude (Anthropic's AI assistant),
    accessed on April 15, 2025.
    https://claude.ai/
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
        
    rng = np.random.default_rng()
    # Define the phase differences for π/4-DQPSK
    phase_differences = np.array([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4])
    
    symbol_indices = rng.integers(0, len(phase_differences), number_of_symbols)
    phase_changes = phase_differences[symbol_indices]
    
    DQPSK_output_symbols = np.zeros(number_of_symbols, dtype='complex')
    DQPSK_output_symbols[0] = np.sqrt(2)/2 * (1 + 1j) # Start symbol
    
    for i in range(1, number_of_symbols):
         # Apply the phase change to the previous symbol
        DQPSK_output_symbols[i] = DQPSK_output_symbols[i-1] * np.exp(1j*phase_changes[i])
    
    DQPSK_symbols_upsampled = np.zeros(number_of_symbols * samples_per_symbol, dtype='complex')
    DQPSK_symbols_upsampled[::samples_per_symbol] = DQPSK_output_symbols

    # Create rectangular pulse shaping filter
    rect_pulse_shape = rectangular_design(samples_per_symbol, normalize=False)
    
    DQPSK_signal = scipy.signal.oaconvolve(DQPSK_symbols_upsampled, rect_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(DQPSK_signal, "Rect pi/4-DQPSK Signal")
    
    return DQPSK_signal

def create_srrc_pi4_dqpsk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates a π/4-DQPSK modulated signal with Square Root Raised Cosine (SRRC) pulse shaping.
    
    Generates a differentially encoded π/4-DQPSK signal using random, IID symbols. 
    The modulation alternates between two QPSK constellations offset by π/4 radians,
    with phase transitions determined by one of four possible phase differences.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the SRRC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the SRRC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband π/4-DQPSK signal after SRRC pulse shaping
        
    π/4-DQPSK implementation based on assistance from Claude (Anthropic's AI assistant),
    accessed on April 15, 2025.
    https://claude.ai/
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
        
    rng = np.random.default_rng()
    # Define the phase differences for π/4-DQPSK
    phase_differences = np.array([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4])
    
    symbol_indices = rng.integers(0, len(phase_differences), number_of_symbols)
    phase_changes = phase_differences[symbol_indices]
    
    DQPSK_output_symbols = np.zeros(number_of_symbols, dtype='complex')
    DQPSK_output_symbols[0] = np.sqrt(2)/2 * (1 + 1j) # Start symbol
    
    for i in range(1, number_of_symbols):
         # Apply the phase change to the previous symbol
        DQPSK_output_symbols[i] = DQPSK_output_symbols[i-1] * np.exp(1j*phase_changes[i])
    
    DQPSK_symbols_upsampled = np.zeros(number_of_symbols * samples_per_symbol, dtype='complex')
    DQPSK_symbols_upsampled[::samples_per_symbol] = DQPSK_output_symbols
    
    SRRC_pulse_shape = srrc_design(samples_per_symbol, filter_span, beta)
    DQPSK_signal = scipy.signal.oaconvolve(DQPSK_symbols_upsampled, SRRC_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(DQPSK_signal, "SRRC pi/4-DQPSK Signal")
        
    return DQPSK_signal

def create_rc_pi4_dqpsk_signal(number_of_symbols, samples_per_symbol, filter_span, beta, plot=False, rng=np.random.default_rng()):
    """
    Creates a π/4-DQPSK modulated signal with Raised Cosine (RC) pulse shaping.
    
    Generates a differentially encoded π/4-DQPSK signal using random, IID symbols. 
    The modulation alternates between two QPSK constellations offset by π/4 radians,
    with phase transitions determined by one of four possible phase differences.
    
    Parameters
    ----------
    number_of_symbols : int
        The number of symbols to generate in the signal
    samples_per_symbol : int
        The number of samples per symbol (oversampling factor)
    filter_span : int
        The number of symbols that the RC filter spans (filter length = filter_span * samples_per_symbol + 1)
    beta : float
        Roll-off factor for the RC filter (0 <= beta <= 1)
    plot : bool
        Plot the signal if True
    rng : numpy Generator
        Symbol sequence is fixed if provided with Generator object with specific seed
    
    Returns
    -------
    numpy.ndarray
        Complex baseband π/4-DQPSK signal after RC pulse shaping
        
    π/4-DQPSK implementation based on assistance from Claude (Anthropic's AI assistant),
    accessed on April 15, 2025.
    https://claude.ai/
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
        
    rng = np.random.default_rng()
    # Define the phase differences for π/4-DQPSK
    phase_differences = np.array([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4])
    
    symbol_indices = rng.integers(0, len(phase_differences), number_of_symbols)
    phase_changes = phase_differences[symbol_indices]
    
    DQPSK_output_symbols = np.zeros(number_of_symbols, dtype='complex')
    DQPSK_output_symbols[0] = np.sqrt(2)/2 * (1 + 1j) # Start symbol
    
    for i in range(1, number_of_symbols):
         # Apply the phase change to the previous symbol
        DQPSK_output_symbols[i] = DQPSK_output_symbols[i-1] * np.exp(1j*phase_changes[i])
    
    DQPSK_symbols_upsampled = np.zeros(number_of_symbols * samples_per_symbol, dtype='complex')
    DQPSK_symbols_upsampled[::samples_per_symbol] = DQPSK_output_symbols
    
    RC_pulse_shape = rc_design(samples_per_symbol, filter_span, beta)
    DQPSK_signal = scipy.signal.oaconvolve(DQPSK_symbols_upsampled, RC_pulse_shape, mode="same")

    # Plot the signal if requested
    if plot:
        plot_signal(DQPSK_signal, "RC pi/4-DQPSK Signal")
        
    return DQPSK_signal