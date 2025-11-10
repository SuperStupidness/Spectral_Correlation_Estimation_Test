import numpy as np
import matplotlib.pyplot as plt

def signal_constellation_generator(modulation_type, plot=False):
    modulation_type = modulation_type.upper()
    
    match modulation_type:
        case "BPSK":
            constellation = np.array([-1, 1])
        case "QPSK":
            constellation = np.array([1+1j, 1-1j, -1-1j, -1+1j]) * np.sqrt(2)/2
        case "8PSK":
            constellation = np.exp(1j*2*np.pi*np.arange(0, 8)/8)
        case "pi/4-DQPSK":
            constellation = np.array([1+1j, 1-1j, -1-1j, -1+1j]) * np.sqrt(2)/2
            pi4_shifted_constellation = constellation * np.exp(1j*np.pi/4)
            constellation = np.concatenate((constellation, pi4_shifted_constellation))
        case "16APSK":
            # Based on DVB-S2, 5.4.3 Bit mapping into 16APSK constellation, Code rate 9/10 specification 
            # Link: https://www.etsi.org/deliver/etsi_en/302300_302399/302307/01.02.01_60/en_302307v010201p.pdf
            APSK_16_inner_ring = np.array([-1-1j, -1+1j, 1+1j, 1-1j]) * 1/2.57
            APSK_16_outer_ring = np.exp(1j*2*np.pi*np.arange(0, 12)/12) * np.exp(1j*np.pi/12)
            constellation = np.concatenate((APSK_16_inner_ring, APSK_16_outer_ring))
        case "32APSK":
            APSK_32_1st_ring = np.array([-1-1j, -1+1j, 1+1j, 1-1j]) * 1/4.3
            APSK_32_2nd_ring = np.exp(1j*2*np.pi*np.arange(0, 12)/12) * np.exp(1j*np.pi/12) * 2.53 * 1/4.3
            APSK_32_3rd_ring = np.exp(1j*2*np.pi*np.arange(0, 16)/16) * np.exp(1j*np.pi/8) 
            constellation = np.concatenate((APSK_32_1st_ring, APSK_32_2nd_ring, APSK_32_3rd_ring))
        case "OOK":
            constellation = np.array([0, 1])
        case modulation_type if modulation_type[-3:] == "QAM":
            number_of_symbols = int(modulation_type[:-3])
            constellation = qam_map_generator(number_of_symbols)
        case modulation_type if modulation_type[-3:] == "ASK":
            number_of_symbols = int(modulation_type[:-3])
            constellation = np.linspace(-1, 1, number_of_symbols)
        case "MSK" | "GMSK":
            raise NameError("Error: Cannot Generate MSK constellation.")
        case modulation_type if modulation_type[-3:] == "FSK":
            raise NameError("Error: Cannot Generate FSK constellation.")
        case _:
            raise NameError(
                "Error: Unrecognizable Modulation Type. Available types: BPSK, QPSK, 8PSK, 16APSK, 32APSK, OOK, M-QAM (16QAM, 32QAM, etc.)")

    if plot:
        _plot_constellation(constellation, f"{modulation_type} constellation")

    return constellation
            
def qam_map_generator(number_of_symbols, plot=False):
    """
    Generate QAM constellation map for any power-of-2 QAM size.
    
    Parameters:
    number_of_symbols (int): QAM size (must be a power of 2)
    plot (bool): Whether to plot the constellation
    
    Returns:
    numpy.ndarray: Complex constellation points
    """
    # Check if number_of_symbols is a power of 2
    if np.mod(np.log2(number_of_symbols), 1) != 0 or number_of_symbols <= 0:
        raise ValueError("Error: Number of symbols must be a power of 2 for QAM")

    # Edge case for 8QAM (square QAM)
    if number_of_symbols == 8:
        constellation = np.array([1, 1+1j, 1j, -1+1j, -1, -1-1j, -1j, 1-1j])
        if plot:
            _plot_constellation(constellation, f"{number_of_symbols}QAM constellation")
            
        return constellation
    
    # For square QAM (e.g., 4, 16, 64, 256)
    if np.mod(np.sqrt(number_of_symbols), 1) == 0:
        # Square root gives the number of points along each axis
        length = int(np.sqrt(number_of_symbols))
        
        # Create evenly spaced grid
        values = np.linspace(-1, 1, length)
        
        # Create 2D grid
        x, y = np.meshgrid(values, values)
        constellation = x + 1j * y
        constellation = constellation.flatten()
    
    # For cross-shaped QAM (e.g., 32, 128, 512)
    else:
        # Determine the length we need to extend on all sides for the cross shape
        inner_square_length = int(np.sqrt(number_of_symbols) / np.sqrt(2))
        remainder = number_of_symbols - inner_square_length**2
        number_of_extra_step = int(remainder/(inner_square_length * 4))
        
        # Determine the next larger square size
        length = inner_square_length + number_of_extra_step * 2
        
        # Get step size 
        values, step = np.linspace(-1, 1, length, retstep=True)
        
        # Creating inner square
        values = values[number_of_extra_step:-number_of_extra_step]
        
        # Create 2D grid
        x, y = np.meshgrid(values, values)
        constellation = x + 1j * y
        constellation = constellation.flatten()
        # print(np.shape(constellation))
        
        # Extending the square
        extended_values = values + 1j
        
        for i in range(1, number_of_extra_step):
            extended_values = np.concatenate((extended_values, values + 1j - i * step * 1j))
            
        extended_values = np.concatenate((extended_values, extended_values * 1j,
                                                  extended_values * (1j)**2, extended_values * (1j)**3))
        
        constellation = np.concatenate((constellation, extended_values))

        if plot:
            _plot_constellation(constellation, f"{number_of_symbols}QAM constellation")
    
    return constellation

def _plot_constellation(constellation, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(constellation.real, constellation.imag, color='blue')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    
    # Add labels
    for i, point in enumerate(constellation):
        plt.text(point.real, point.imag, str(i+1), fontsize=8)
        
    plt.title(title)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.xticks(np.linspace(-1, 1, num=5))
    plt.yticks(np.linspace(-1, 1, num=5))
    plt.show()
    
    return