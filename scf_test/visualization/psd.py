import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def psd(signal, fs=1, method='daniell', L=256, db=False, plot=False):
    N = len(signal)

    match method.lower():
        case 'daniell':
            if L > N:
                # Let filter be 5% of the signal length
                L = int(N * 0.02)
            
            power_spectrum = 1/(N*fs) * np.abs(np.fft.fftshift(np.fft.fft(signal)))**2
        
            # Moving average filter 
            # Ensure filter is odd length
            if L % 2 == 0:
                L = L + 1
        
            g = np.ones(L)/L
        
            # Smoothing to obtain PSD
            psd = scipy.signal.oaconvolve(power_spectrum, g)
        
            # Remove filter transient
            # remove L-1 samples on the left and extend (L-1)/2 using left most sample
            # repeat on the right
            start = int((L-1)/2)
            psd_truncated = np.zeros(N)
            psd_truncated[:start] = psd[L-1] 
            psd_truncated[start:-start] = psd[L-1:-(L-1)]
            psd_truncated[-start:] = psd[-(L-1)]

            psd = psd_truncated
            
        case 'bartlett':
            power_sum = np.zeros(L)
            num_time_segments = int(np.ceil(N/L))
        
            for i in range(num_time_segments):
                power_spectrum = 1/(L*fs) * np.abs(np.fft.fftshift(np.fft.fft(signal[i*L:min((i+1)*L, N)], n=L)))**2
      
                power_sum += power_spectrum
        
            psd = power_sum/num_time_segments
            
        case 'welch':
            power_sum = np.zeros(L)
            num_time_segments = int(np.ceil(2*N/L))
        
            for i in range(num_time_segments):
                power_spectrum = 1/(L*fs) * np.abs(np.fft.fftshift(np.fft.fft(signal[int(i*L/2):min(int((i+2)*L/2), N)], n=L)))**2
                    
                power_sum += power_spectrum
        
            psd = power_sum/num_time_segments
        case _:
            raise Exception("Unknown PSD method. Supported method: 'daniell'(default), 'barlett', 'welch'")
            return

    if db:
        psd = 10*np.log10(psd + 1e-10)

    if plot:
        f = np.linspace(-fs/2, fs/2, len(psd))
        plt.xlabel("Frequency (Hz)")
        plt.plot(f, psd)
        
        if db:
            plt.ylabel("Power (dB)")
        else:
            plt.ylabel("Power (W/Hz)")
            
        plt.title(f"Power Spectral Density ({method}'s method)")
        
    return psd

def db2mag(signal_db):
    signal_mag = 10**(signal_db/10)

    return signal_mag


        


    