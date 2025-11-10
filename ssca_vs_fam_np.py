import pyfftw
import pytest
from matplotlib import pyplot as plt
import numpy as np

def ssca(signal, fs=1, L=64, conjugate=False, plot=False, coherence=False):   
    # Pad signal to make it even -> Faster fft
    N = len(signal)
    if N % 2 == 1:
        N = N + 1
        signal = np.pad(signal, (0, 1))

    if L % 2 == 0:
        pad_length = int(L/2)
    else:
        pad_length = int((L-1)/2)

    # Initialize hamming windows

    window = np.hamming(L).reshape((-1, 1))

    window_2 = np.hamming(N)
    
    window_2 = window_2/np.sum(window_2)

    conj_signal = np.conj(signal)

    # Pad signal for windowing
    signal_padded = np.pad(signal, (pad_length, pad_length))

    # Initialize memory aligned variable -> Faster fft
    Xt = pyfftw.empty_aligned((L, N), dtype='complex')

    # Channelizer
    for i in range(L):
        Xt[i, :] = signal_padded[i:i+N]

    # Windowing
    Xt = Xt * window

    # FFT
    Xt = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=0), axes=0)

    # Enable cache for pyfftw (faster fft)
    pyfftw.interfaces.cache.enable()
          
    # Exponential to calculate complex demodulate
    k = np.linspace(-L/2, L/2 - 1, L).reshape((-1,1))
    n = np.linspace(0, N - 1, N)
    E = np.exp(-1j*2*np.pi*n*k/L)
    
    # Multiply with original conjugated signal and second FFT
    if not conjugate:
        Xt = Xt * conj_signal * E * window_2
    else:
        Xt = Xt * signal * E * window_2

    ssca = np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=1), axes=1))

    # Map spectral and cycle frequencies to SSCA output
    q = np.linspace(-N/2, N/2 - 1, N)
    f = k/(2*L) - q/(2*N)
    alpha = k/L + q/N

    # Optional spectral coherence calculation from SSCA SCF
    if coherence:
        sample_1 = np.rint((f + alpha/2) * L/fs).astype(int)
        if conjugate:
            sample_2 = np.rint((alpha/2 - f) * L/fs).astype(int)
        else:
            sample_2 = np.rint((f - alpha/2) * L/fs).astype(int)

        smoothing_width = int(0.02 * N) # 2% sampling rate

        psd_fsm = np.fft.ifftshift(psd(signal, L=L, method="bartlett", db=False, plot=False))

        coherence_denominator = np.sqrt(psd_fsm[sample_1] * psd_fsm[sample_2])

        ssca_coh = ssca/coherence_denominator

    # Optional plot option (Not very good)
    if plot:
        plt.figure(figsize=(6, 6))

        # FSM method for visual only
        # ssca_smooth = scipy.signal.oaconvolve(ssca, 1/256 * np.ones((1, 256)), mode = "same")

        if coherence:
            plt.pcolormesh(f, alpha, ssca_coh)
        else:
            plt.pcolormesh(f, alpha, ssca)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Cycle Frequency (Hz)")
        if conjugate:
            plt.title("Conj SSCA")
        else:
            plt.title("SSCA")

    if coherence:
        return ssca_coh, f, alpha 
    else:
        return ssca, f, alpha 

def fam(signal, Np=64, L=16, conjugate=False, plot=False, coherence=False, fs=1):
    # L check
    if L < 1:
        L = 1
    
    # Pad signal (ensure that it is even -> fast fft)
    N = len(signal)
    if N % 2 == 1:
        N = N + 1
        signal = np.pad(signal, (0, 1))

    if Np % 2 == 1:
        Np = Np + 1

    # Pad signal more due to hopping
    P = int(np.floor(N / L))

    pad_length = int(Np - (N - P * L))

    pad_length = int(pad_length/2)
    
    signal_padded = np.pad(signal, (pad_length, pad_length))

    # Normalize energy and power on window to get correct PSD magnitude
    window = np.hamming(Np).reshape((-1, 1))

    window = window/ np.sqrt(np.sum(window**2))

    window_2 = np.hamming(P)
    
    window_2 = window_2 / np.sum(window_2)

    N = len(signal_padded)

    L = int(L)

    Xt = pyfftw.empty_aligned((Np, P), dtype='complex')

    # Channelize signal with hop (Put segments in rows)
    for i in range(Np):
        Xt[i, :] = signal_padded[i:i + P*L:L]

    # Apply windowing function
    Xt = Xt * window

    # Enable cache for pyfftw (faster fft)
    pyfftw.interfaces.cache.enable()

    # FFT + Phase compensation
    Xt = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=0), axes=0)

    fk = np.fft.fftshift(np.fft.fftfreq(Np))

    i = np.arange(P)

    phase_compensation = np.exp(-1j*2*np.pi*fk.reshape((-1,1))*i*L)

    # Forming Yt*
    if conjugate:
        Xt = Xt * phase_compensation
        
        Yt_conj = np.roll(np.flipud(np.conj(Xt)), 1, axis=0)
        Yt_conj = np.conj(Yt_conj)
        
    else:
        Xt = Xt * phase_compensation 
        Yt_conj = np.conj(Xt)

    # Frequency and alpha mapping calculation
    # alpha_i = fk - fl
    # alpha = alpha_i + q * delta_alpha = alpha_i + q/N
    # fj = (fk + fl)/2 = (k + l)/(2*L)
    
    fl = np.tile(fk.reshape((-1,1)), P)

    # Retain q in this range to minimize variability near top and bottom of channel pair region
    q_lim = int(P*L/(2*Np))

    q_size = q_lim*2

    q = np.arange(q_size) - q_lim
    delta_alpha = q/N

    fam = pyfftw.empty_aligned((Np * Np, q_size), dtype='float')
    fj = np.zeros((Np * Np, q_size), dtype='float')
    alpha = np.zeros((Np * Np, q_size), dtype='float')

    l = np.tile(np.arange(Np).reshape((-1, 1)), q_size) 

    # Multiply rows of Xt to Yt* and window + fft
    for k in range(Np):
        fj[k*Np:(k+1)*Np, :] = (k + l - Np)/(2*Np) 
        fam[k*Np:(k+1)*Np, :] = np.abs(pyfftw.interfaces.numpy_fft.fft(Xt[k, :] * Yt_conj * window_2, axis=1))[:, q]
        alpha[k*Np:(k+1)*Np, :] = (k - l)/Np + delta_alpha 

    # Remove any value outside the principal diamond
    mask = np.abs(fj) + np.abs(alpha/2) > 0.5

    fam[mask] = 0

    fam = fam.reshape((Np, Np*q_size))
    fj = fj.reshape((Np, Np*q_size))
    alpha = alpha.reshape((Np, Np*q_size))
    
    # Optional spectral coherence calculation from SSCA SCF
    if coherence:
        N = Np
        
        sample_1 = np.rint((fj + alpha/2) * N/fs).astype(int)
        if conjugate:
            sample_2 = np.rint((alpha/2 - fj) * N/fs).astype(int)
        else:
            sample_2 = np.rint((fj - alpha/2) * N/fs).astype(int)

        smoothing_width = int(0.02 * N) # 15% sampling rate

        psd_fsm = np.fft.ifftshift(psd(signal, L=Np, method="bartlett", db=False, plot=False))

        coherence_denominator = np.sqrt(psd_fsm[sample_1] * psd_fsm[sample_2])

        fam_coh = fam/coherence_denominator

    if plot:
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(fj, alpha, fam)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Cycle Frequency (Hz)")
        if conjugate:
            plt.title("Conj FAM")
        else:
            plt.title("FAM")

    if coherence:
        return fam_coh, fj, alpha 
    else:
        return fam, fj, alpha

def fam_2(signal, Np, L, conjugate=False, plot=False, coherence=False, fs=1):
    # L check
    if L < 1:
        L = 1
    
    # Pad signal (ensure that it is even -> fast fft)
    N = len(signal)
    if N % 2 == 1:
        N = N + 1
        signal = np.pad(signal, (0, 1))

    if Np % 2 == 1:
        Np = Np + 1

    # Pad signal more due to hopping
    P = int(np.floor(N / L))

    pad_length = int(Np - (N - P * L))

    pad_length = int(pad_length/2)
    
    signal_padded = np.pad(signal, (pad_length, pad_length))

    # Normalize energy and power on window to get correct PSD magnitude
    window = np.hamming(Np).reshape((-1, 1))

    window = window/ np.sqrt(np.sum(window**2))

    window_2 = np.hamming(P)
    
    window_2 = window_2 / np.sum(window_2)

    N = len(signal_padded)

    L = int(L)

    Xt = pyfftw.empty_aligned((Np, P), dtype='complex')

    # Channelize signal with hop
    for i in range(Np):
        Xt[i, :] = signal_padded[i:i + P*L:L]

    # Apply windowing function
    Xt = Xt * window

    Xt_conj = np.conj(Xt)

    # Enable cache for pyfftw (faster fft)
    pyfftw.interfaces.cache.enable()

    # FFT + Phase compensation
    Xt = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=0), axes=0)

    fk = np.fft.fftshift(np.fft.fftfreq(Np))

    i = np.arange(P)

    phase_compensation = np.exp(-1j*2*np.pi*fk.reshape((-1,1))*i*L)

    # Forming Yt*
    if conjugate:
        Yt_conj = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt_conj, axis=0), axes=0)
        Yt_conj = np.conj(Yt_conj * phase_compensation)
        Xt = Xt * phase_compensation 
    else:
        Xt = Xt * phase_compensation 
        Yt_conj = np.conj(Xt)

    # Frequency and alpha mapping calculation
    # alpha_i = fk - fl
    # alpha = alpha_i + q * delta_alpha = alpha_i + q/N
    # fj = (fk + fl)/2 = (k + l)/(2*L)
    
    fl = np.tile(fk.reshape((-1,1)), P)

    # Retain q in this range to minimize variability near top and bottom of channel pair region
    q_lim = int(P*L/(2*Np))

    q_size = q_lim*2

    q = np.arange(q_size) - q_lim
    delta_alpha = q/N

    fam = pyfftw.empty_aligned((Np * Np, q_size), dtype='float')
    fj = np.zeros((Np * Np, q_size), dtype='float')
    alpha = np.zeros((Np * Np, q_size), dtype='float')

    l = np.tile(np.arange(Np).reshape((-1, 1)), q_size) 

    # Multiply rows of Xt to Yt* and window + fft
    for k in range(Np):
        fj[k*Np:(k+1)*Np, :] = (k + l - Np)/(2*Np) 
        fam[k*Np:(k+1)*Np, :] = np.abs(pyfftw.interfaces.numpy_fft.fft(Xt[k, :] * Yt_conj * window_2, axis=1))[:, q]
        alpha[k*Np:(k+1)*Np, :] = (k - l)/Np + delta_alpha 

    # Remove any value outside the principal diamond
    # mask = np.abs(fj) + np.abs(alpha/2) > 0.5

    #fam[mask] = 0

    fam = fam.reshape((Np, Np*q_size))
    fj = fj.reshape((Np, Np*q_size))
    alpha = alpha.reshape((Np, Np*q_size))
    
    # Optional spectral coherence calculation from SSCA SCF
    if coherence:
        N = Np
        
        sample_1 = np.rint((fj + alpha/2) * N/fs).astype(int)
        if conjugate:
            sample_2 = np.rint((alpha/2 - fj) * N/fs).astype(int)
        else:
            sample_2 = np.rint((fj - alpha/2) * N/fs).astype(int)

        smoothing_width = int(0.02 * N) # 15% sampling rate

        psd_fsm = np.fft.ifftshift(psd(signal, L=Np, method="bartlett", db=False, plot=False))

        coherence_denominator = np.sqrt(psd_fsm[sample_1] * psd_fsm[sample_2])

        fam_coh = fam/coherence_denominator

    if plot:
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(fj, alpha, fam)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Cycle Frequency (Hz)")
        if conjugate:
            plt.title("Conj FAM")
        else:
            plt.title("FAM")

    if coherence:
        return fam_coh, fj, alpha 
    else:
        return fam, fj, alpha

N = 2**15 #32768

@pytest.mark.parametrize("Np", [
    (8),
    (16),
    (32),
    (64),
    (128),
    (256),
    (512)
])

def test_ssca(benchmark, Np):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(ssca, signal, L=Np, conjugate=False)


@pytest.mark.parametrize("Np", [
    (8),
    (16),
    (32),
    (64),
    (128),
    (256),
    (512)
])

def test_fam(benchmark, Np):
    # Setup runs once before all benchmark iterations
    rng = np.random.default_rng()
    signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
    
    # Benchmark the function
    result = benchmark(fam_2, signal, Np=Np, L=Np/4, conjugate=False)
