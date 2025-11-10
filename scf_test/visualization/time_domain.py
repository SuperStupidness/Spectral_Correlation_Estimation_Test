import numpy as np
import matplotlib.pyplot as plt

def plot_signal(signal, title, max_samples=500):
    """Plot the I/Q components of a signal"""
    
    # Get the first 500 samples or the entire signal if shorter
    plot_length = min(max_samples, len(signal))
    
    # Extract I/Q components
    i_component = np.real(signal[:plot_length])
    q_component = np.imag(signal[:plot_length])
    
    # Create plot
    plt.figure(figsize=(4.8, 3))
    plt.plot(i_component, 'b-', label='In-Phase (I)')
    plt.plot(q_component, 'r-', label='Quadrature (Q)')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

def plot_fsk_signal(signal, symbols, phi, delta_f, f_carrier, title, max_samples=200):
    """Plot the real component and instantanteous frequency of a FSK signal"""
    t = np.linspace(0, len(phi), len(phi))

    inst_freq = np.zeros(len(t))
    inst_freq[1:] = np.diff(phi) / (2 * np.pi * (t[1] - t[0]))
    inst_freq[0] = inst_freq[1]  # Avoid undefined first point

    ylim_upper = f_carrier + delta_f * 1.1 
    ylim_lower = f_carrier - delta_f * 1.1  

    fig, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t[:max_samples], np.real(signal[:max_samples]))
    ax1.plot(symbols[:max_samples])
    ax1.set_title(title)
    ax1.set_ylabel("Magnitude")
    ax1.legend(["Real", "Bit symbol"])
    
    ax2.plot(inst_freq[:max_samples], color="red") # To check whether it is coherent or not
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Frequency")
    ax2.set_ylim([ylim_lower, ylim_upper])
    ax2.legend(["Frequency"])
    
    plt.show()
    return