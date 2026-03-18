import numpy as np
import pyfftw
import matplotlib.pyplot as plt
import scipy.signal 

def ssca(signal, fs=1, Np=64, conjugate=False, plot=False, coherence=False):   
    # Pad signal to make it even -> Faster fft
    N = len(signal)
    if N % 2 == 1:
        N = N + 1
        signal = np.pad(signal, (0, 1))

    if Np % 2 == 0:
        pad_length = int(Np/2)
    else:
        pad_length = int((Np-1)/2)

    # Initialize hamming windows

    window = np.hamming(Np).reshape((-1, 1))

    window_2 = np.hamming(N)
    
    window_2 = window_2/np.sum(window_2)

    conj_signal = np.conj(signal)

    # Pad signal for windowing
    signal_padded = np.pad(signal, (pad_length, pad_length))

    # Initialize memory aligned variable -> Faster fft
    Xt = pyfftw.empty_aligned((Np, N), dtype='complex')

    # Channelizer
    for i in range(Np):
        Xt[i, :] = signal_padded[i:i+N]

    # Windowing
    Xt = Xt * window

    # FFT
    Xt = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=0), axes=0)

    # Enable cache for pyfftw (faster fft)
    pyfftw.interfaces.cache.enable()
          
    # Exponential to calculate complex demodulate
    k = np.linspace(-Np/2, Np/2 - 1, Np).reshape((-1,1))
    n = np.linspace(0, N - 1, N)
    E = np.exp(-1j*2*np.pi*n*k/Np)
    
    # Multiply with original conjugated signal and second FFT
    if not conjugate:
        Xt = Xt * conj_signal * E * window_2
    else:
        Xt = Xt * signal * E * window_2

    ssca = np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=1), axes=1))

    # Map spectral and cycle frequencies to SSCA output
    q = np.linspace(-N/2, N/2 - 1, N)
    f = k/(2*Np) - q/(2*N)
    alpha = k/Np + q/N

    # Optional spectral coherence calculation from SSCA SCF
    if coherence:
        sample_1 = np.rint((f + alpha/2) * Np/fs).astype(int)
        if conjugate:
            sample_2 = np.rint((alpha/2 - f) * Np/fs).astype(int)
        else:
            sample_2 = np.rint((f - alpha/2) * Np/fs).astype(int)

        psd_fsm = np.fft.ifftshift(psd(signal, L=Np, method="bartlett", db=False, plot=False))

        coherence_denominator = np.sqrt(psd_fsm[sample_1] * psd_fsm[sample_2])

        ssca_coh = ssca/coherence_denominator

    # Optional plot option (Not very good)
    if plot:
        plt.figure(figsize=(6, 6))

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
    
def Fast_SC(x,Nw,alpha_max,Fs,opt, WindowType = 'hann'):

    # -->===================================
    # -->Check inputs
    # -->===================================
    if (alpha_max > Fs/2):
        print('\'alpha_max\' must be smaller than Fs/2!')
    
    if (alpha_max < 0):
        print('\'alpha_max\' must be non-negative!')
 
    # -->===================================
    # -->Set value of overlap
    # -->===================================
    Nv,dt,da,df = param_Fast_SC( len(x), Nw, alpha_max,Fs)
    
    # -->===================================
    # -->Computation of short-time Fourier transform 
    # -->===================================
    
    # Enable cache for pyfftw (faster fft)
    pyfftw.interfaces.cache.enable()
  
    # STFT,f,t = LiteSpectrogram(x,Nw,Nv,Nw,Fs)
    STFT,f,t = LiteSpectrogram(x = x, Window = Nw, Noverlap = Nv, Nfft= Nw,Fs = Fs, WinType = WindowType)

    # -->===================================
    # -->Fast spectral correlation/coherence
    # -->===================================
    S, alpha, __ ,__ =   Fast_SC_STFT( STFT = STFT, Dt = dt, Wind = Nw, opt = opt, Fs = Fs,  Nfft = np.array([]), WinType = WindowType)

    I = np.where(alpha <= alpha_max)
    I = []
    I.extend( i for i in range(0,len(alpha)) if(alpha[i] <= alpha_max) )
    alpha = alpha[I[:]]
    S = S[:,I]  

    return(S,alpha,f,STFT,t,Nv)

def param_Fast_SC(L,Nw,alpha_max,Fs):

    # -->===================================    
    # -->block shift
    # -->===================================
    
    R = np.fix(Fs/2/alpha_max)
    R = np.maximum(1, np.minimum(R, np.fix(.25*Nw) ) )

    # -->===================================
    # -->block overlap
    # -->===================================
    Nv = Nw - R
    
    # -->===================================
    # -->time resolution of STFT (in s)
    # -->===================================    
    dt = R/Fs

    # -->===================================    
    # -->cyclic frequency resolution (in Hz)
    # -->===================================    
    da = Fs/L

    # -->===================================    
    # -->carrier frequency resolution (in Hz)
    # -->===================================
    df = Fs/Nw*np.sum(Nw**2)/np.mean(Nw)**2
    
    
    return(Nv,dt,da,df)

def Fast_SC_STFT( STFT, Dt, Wind, opt, Fs = 1, Nfft = None, WinType = 'hann' ):

    NF = STFT.shape[0]    
    Nw = 2*(NF-1)      # -->window length
    
    flag = 0
    
    if Nfft is None:
        Nfft = STFT.shape[1]
    else:
        if (Nfft.size == 0):
            Nfft = STFT.shape[1]
    
    
    if (opt == None):
        opt = { "abs": 0, "calib": 1, "coh": 0}
    else:
        if not("abs" in opt):
            opt["abs"] = 0
    
        if not("calib" in opt):
            opt["calib"] = 1
    
        if not("coh" in opt):
            opt["coh"] = 0
    
    if(np.size(Wind) == 1):
        Wind = GetWindow(WinType, Wind)



        # Window = io.loadmat("D:\GDRB_COC\codes\src_python\lib\matlab_hanning_window.mat")
        # Wind = Window['Window']
    # -->===================================
    # -->Whitening the STFT for computing the spectral coherence
    # -->===================================
    if(opt["coh"] == 1):
        Sx = np.mean( abs(STFT)**2, axis=1)  # -->Mean power spectral density
        Sx = Sx.reshape((Sx.shape[0],1))
        termo1 = 1/np.sqrt(Sx)
        termo2 = np.tile( termo1, (1, STFT.shape[1]) )
        STFT = np.multiply(STFT,termo2)
        # STFT = np.multiply( STFT, np.tile( 1/np.sqrt(Sx), (1, STFT.shape[1]) ) )
    # -->===================================
    # -->Computation of the cyclic modulation spectrum
    # -->===================================
    S,alpha, __,__,__ = CPS_STFT_zoom(0,STFT,Dt,Wind,Fs,Nfft,flag)
    W0,__, __ = Window_STFT_zoom(alpha,0,Dt,Wind,Nfft,'full', Fs = Fs)
    # W0,__, __ = Window_STFT_zoom(alpha,0,Dt,Wind,Nfft,'trunc', Fs = Fs)
    
    if (opt["abs"] == 1):
        S = np.abs(S)
        W = np.array(W0, copy=True)
        W = np.abs(W)
    else:
        W = np.array(W0, copy=True)
  
    # for i in range(int(np.ceil(Nfft/2)+1),Nfft):
    #     aux = W0[i]
    #     W[i] = 0
    #     W0[i]=aux
    W[int(np.ceil(Nfft/2)+1):Nfft] = 0 # -->truncate negative frequencies
    
    # -->===================================
    # -->Number of scans
    # -->===================================
    Fa = 1/Dt               # -->cyclic sampling frequency in Hz
    K = np.fix(Nw/2*Fa/Fs)


    
    for k in range(1,int(K)): 

        # -->===================================
        # -->positive cyclic frequencies
        # -->===================================
    
        Stemp,alpha,alpha0,__,__ = CPS_STFT_zoom(k/Nw*Fs,STFT,Dt,Wind,Fs,Nfft,flag)
            
        Wtemp = Shift_Window_STFT_zoom(W0, alpha0/Fa*Nfft,'trunc')
        # -->Wtemp = Window_STFT_zoom(alpha,alpha0,Dt,Wind,Nfft,Fs,'trunc')
        if(opt["abs"] == 1):
            S[:,2:Nfft] = S[:,2:Nfft] + abs(Stemp[:,2:Nfft])
            W[2:Nfft] = W[2:Nfft] + abs(Wtemp[2:Nfft])
        else:

            # W = W.reshape((W.shape[1],1))
            S[:,2:Nfft] = S[:,2:Nfft] + Stemp[:,2:Nfft]
            W[2:Nfft] = W[2:Nfft] + Wtemp[2:Nfft]
        # -->plot(abs(Wtemp),':'),plot(W)
        
        # -->negative cyclic frequencies
        Stemp,alpha,alpha0,__,__ = CPS_STFT_zoom(-k/Nw*Fs,STFT,Dt,Wind,Fs,Nfft,flag)
        Wtemp = Shift_Window_STFT_zoom(W0,alpha0/Fa*Nfft,'trunc')
        # -->Wtemp = Window_STFT_zoom(alpha,alpha0,Dt,Wind,Nfft,Fs,'trunc')
        if (opt["abs"] == 1):
            S[:,2:Nfft] = S[:,2:Nfft] + abs(Stemp[:,2:Nfft])
            W[2:Nfft] = W[2:Nfft] + abs(Wtemp[2:Nfft])
        else:
            S[:,2:Nfft] = S[:,2:Nfft] + Stemp[:,2:Nfft]
            W[2:Nfft] = W[2:Nfft] + Wtemp[2:Nfft]

    # -->===================================
    # -->Calibration
    # -->===================================
            
    if (opt["calib"] == 1):
        Winv = np.ones((Nfft,1))
        I,__ = np.where(W < .5*W[0])
        Winv[0:I[0]] = 1/W[0:I[0]]
        Winv[I[0]+1:Nfft] = 1/W[I[0]+1]
        valor_aux1 = np.sum(Wind[:]**2)
        valor_aux2 = np.tile(Winv.transpose(),(NF,1))
         
        S = np.multiply( S, valor_aux2*valor_aux1)
        # S = S*np.tile(Winv.transpose(),(NF,1))*np.sum(Wind[:]**2)
        
    else:
        Winv = 1/W(0)
        S = S*Winv*sum(Wind[:]**2)
    # -->===================================
    # -->Impose real values at zero cyclic frequency
    # -->===================================
    # -->S(:,1) = real(S(:,1))
    if(opt["coh"]== 1):
        valor_aux3 = np.mean(S[:,0])
        S = S/valor_aux3
        

    return(S,alpha,W,Winv)
    # -->Subroutines of Fast_SC_STFT.m
    
def CPS_STFT_zoom(alpha0, STFT, Dt, Window, Fs = 1, Nfft = [], flag = 0):


    
    # [NF,NT,N3] = STFT.shape
    NF,NT = STFT.shape
    N3 = 1
    # [NF,NT,N3] = np.size(STFT)
    Nw = 2*(NF-1)          # --> window length
    Fa = 1/Dt    # --> cyclic sampling frequency in Hz

    if(np.size(Nfft) == 0):
        Nfft = NT
    else:
        if(Nfft < NT):
            raise ValueError('Nfft must be greater than or equal to the number of time samples in STFT!')
    
    # -->===================================
    # --> Check for aliasing
    # -->===================================
            
    if(flag == 0):
        if (np.abs(alpha0) > Fa/2):
            print(f'|alpha0| must be selected smaller than {Fa/2} !!')


    # -->===================================
    # --> Vector of cyclic frequencies
    # -->===================================

    alpha = np.arange(0,Nfft)/Nfft*Fa
 
    # -->===================================
    # --> Computation "cross-frequency" cyclic modulation spectrum
    # -->===================================
    fk = int(np.round(alpha0/Fs*Nw))
    alpha0 = fk/Nw*Fs

    if(N3 == 1):
        if(fk >= 0):
            S = np.multiply( 
                                np.vstack( (STFT[fk:NF , : ] ,  np.zeros((fk,NT))) ),
                                STFT.conjugate()
                            )
        else:
            S = np.multiply(
                                np.vstack( (STFT[-fk:NF , : ].conjugate() , np.zeros((-fk,NT)))  ),
                                STFT
                            )

    else:
        if( fk >= 0):
            S = np.multiply(
                                np.vstack( ( np.squeeze( STFT[fk:NF,:,1]) , np.zeros(fk,NT) ) ),
                                np.squeeze(STFT[:,:,2]).conjugate()
                            )
        else:
            S = np.multiply(
                                np.vstack( ( np.squeeze(STFT[-fk:NF,:,1]).conjugate() , np.zeros(-fk,NT) ) ),
                                np.squeeze(STFT[:,:,2])
                            )
    # S = np.fft.fft(S, n=Nfft)/NT
    S = pyfftw.interfaces.numpy_fft.fft(S, n=Nfft, axis=1)/NT

    # -->===================================
    # --> Calibration
    # -->===================================
    valor_aux = np.sum(Window[:]**2)
    valor_aux = valor_aux/Fs
    S = S/valor_aux
    # S = S/np.sum(Window[:]**2)/Fs

    # -->===================================
    # --> Removal of aliased cyclic frequencies
    # -->===================================
    ak = np.round(alpha0/Fa*Nfft)
    valor_aux = np.arange(int(np.ceil(Nfft/2)+1+ak),Nfft)
    S[:,int(np.ceil(Nfft/2)+1+ak):Nfft] = 0
    # S[:,int(np.ceil(Nfft/2)+ak):Nfft] = 0

    # -->===================================
    # --> Phase correction
    # -->===================================

    Iw = np.argmax(Window)

    a2 = alpha - alpha0
    a2 = a2/Fs
    a2 = -2j*np.pi*Iw*a2
    a2 = np.exp(a2)
    S = np.multiply(    
                        S,
                        np.tile(a2, (NF,1))
                    )

    return(S,alpha,alpha0,fk,Fa)

def Window_STFT_zoom(alpha,alpha0,Dt,Window,Nfft, opt, Fs = 1):

    Fa = 1/Dt    # --> cyclic sampling frequency in Hz

    # -->===================================
    # --> Computation the "zooming" window
    # -->===================================
    WSquared = Window[:]**2
    Iw = np.argmax(Window) # --> set origin of time to the centre of symmetry (maximum value) of the window
    W1 = np.zeros((Nfft,1))
    W2 = np.zeros((Nfft,1))
    
    # n = np.arange(0,Iw)
    n = np.arange(1,Iw+1).reshape((Iw,1))
    n = n/Fs


    # plt.figure()
    # plt.plot(WSquared)
    
    # fig, ([ax1,ax2],[ax3,ax4], [ax5, ax6]) = plt.subplots(3,2)

    for k in range(0,Nfft):

        # -->===================================
        # --> "positive" frequencies
        # -->===================================
        T =  WSquared[Iw:0:-1].reshape((WSquared[Iw:0:-1].shape[0],1))
        # valor1 = (2*np.pi*n*(alpha[k]-alpha0))
        valor1 = np.cos(2*np.pi*n*(alpha[k]-alpha0))
        valor2 = ( np.multiply( T, valor1 ))
        # valor2 = ( np.dot( WSquared[Iw:0:-1], valor1 ))
        # valor2 = ( WSquared[Iw:0:-1]*valor1 )
        valor3 = 2*np.sum( valor2)
        # valor2 = 2*np.sum( WSquared[Iw:0:-1]*valor1)
        W1[k] = WSquared[Iw] + valor3

        # W1[k] = WSquared[Iw] + 2*np.sum( np.multiply( WSquared[Iw:0:-1], np.cos(2*np.pi*n*(alpha[k]-alpha0)) ))

        # -->===================================
        # --> "negative" frequencies (aliased)
        # -->===================================
        valor4 = np.cos(2*np.pi*n*(alpha[k]-alpha0-Fa))
        valor5 =  np.multiply(T,valor4 )
        # valor5 =  np.dot(WSquared[Iw:0:-1],valor4 )
        valor6 = 2*np.sum(valor5 )
        W2[k] = WSquared[Iw] + valor6
        # W2[k] = WSquared[Iw] + 2*np.sum( np.multiply( WSquared[Iw:0:-1], np.cos(2*np.pi*n*(alpha[k]-alpha0-Fa))) )
        
        # W2(k) = WSquared(Iw) + 2*sum(WSquared(Iw-1:-1:1).*cos(2*pi*n*(alpha(k)-alpha0-Fa)));

        # ax1.plot(W1)
        # ax3.plot(valor1)
        # ax5.plot(valor2)
        # # ax3.plot(valor3)
        # ax2.plot(W2)
        # ax4.plot(valor4)
        # ax6.plot(valor5)
        
        
    W = W1 + W2
    # -->===================================
    # --> Note: sum(W2) = max(W)
    # -->===================================

    # -->===================================
    # --> Removal aliased cyclic frequencies
    # -->===================================

    if 'trunc' in opt:
        ak = np.round(alpha0/Fa*Nfft)
        W[int(np.ceil(Nfft/2)+1+ak):Nfft] = 0

    return (W,W1,W2)

def Shift_Window_STFT_zoom(W0,a0,opt):

    Nfft = len(W0)
    
    # -->===================================
    # --> Circular shift with linear interpolation for non-integer shifts
    # -->===================================
    a1 = int(np.floor(a0))
    a2 = int(np.ceil(a0))

    if (a1 == a2):
        # W = np.roll(W0,a0,axis=0)
        W = np.roll(W0,int(a0),axis=0)
    else:
        valor_aux = a0-a1
        valor_aux = np.roll(W0,a2,axis=0)*valor_aux
        W = np.roll(W0,a1,axis=0)*(1-(a0-a1)) + valor_aux
        

        # W = np.roll(W0,a1)*(1-(a0-a1)) + np.roll(W0,a2)*(a0-a1)
    
    # -->===================================
    # --> Removal of aliased cyclic frequencies
    # -->===================================
    if 'trunc' in opt:
        W[ int(np.ceil(Nfft/2)+1+round(a0)):Nfft] = 0
    
    return(W)

def LiteSpectrogram(x, Window, Noverlap,Nfft,Fs = 1, WinType = 'hann'):

    from scipy import signal
    
    if (np.size(Window) == 1):
        Window = GetWindow(WinType, Window)

        
    Window = Window[:]
    n = len(x)                  # --> Number of data points
    nwind = len(Window)         # --> length of window
    R = nwind - Noverlap        # --> block shift
    x = x[:]		
    K = np.fix((n-Noverlap)/(nwind-Noverlap))	# --> Number of windows
                                                
    # -->===================================
    # --> compute STFT
    # -->===================================
    index0 = 0
    index1 = nwind
   
    X = np.zeros( (int((Nfft/2)+1),int(K)), dtype='complex128' )
    for k in range(0, int(K)):

        
        Xw = pyfftw.interfaces.numpy_fft.fft( np.multiply(Window, x[int(index0):int(index1)]), n=Nfft)		# --> Xw(f-a/2) or Xw(f-a)

        X[:,k] = Xw[0:int(Nfft/2+1)]
        index0 = index0+R
        index1 = index1+R
        
    f = np.arange(0,int(Nfft/2)+1)/Nfft*Fs
    t = np.arange(nwind/2, nwind/2+(K-1)*R, R)/Fs

    return (X, f, t)




def GetWindow(window_type, window_size):
    if window_type == 'hann':
        Window = scipy.signal.get_window("hann", window_size)
    elif window_type == 'hamming':
        Window = scipy.signal.get_window("hamming", window_size)
    elif window_type == 'blackman':
        Window = scipy.signal.get_window("blackman", window_size)
    elif window_type == 'kaiser':
        Window = scipy.signal.get_window(("kaiser", 15), window_size )
    elif window_type == 'gaussian':
         Window = scipy.signal.get_window(("gaussian", 80), window_size)
    elif window_type == 'chebwin':
        Window = scipy.signal.get_window(("chebwin", 80), window_size)
    else:
        Window = scipy.signal.get_window("hann", window_size)
    return(Window)


def fast_sc_wrapper(signal, Np=64, conjugate=False, coherence=False):
    if coherence:
        opt = {"abs": 0, "calib": 1, "coh": 1}
    else:
        opt = {"abs": 0, "calib": 1, "coh": 0}

    Sx, alpha, f, _, _, _ = Fast_SC(signal, Nw=Np, alpha_max=0.5, Fs=1, opt=opt)

    alpha, f = np.meshgrid(alpha, f)

    return np.abs(Sx), f, alpha

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
            psd = signal.oaconvolve(power_spectrum, g)
        
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

def ssca_flattop(signal, fs=1, Np=64, conjugate=False, plot=False, coherence=False):   
    # Pad signal to make it even -> Faster fft
    N = len(signal)
    if N % 2 == 1:
        N = N + 1
        signal = np.pad(signal, (0, 1))

    if Np % 2 == 0:
        pad_length = int(Np/2)
    else:
        pad_length = int((Np-1)/2)

    # Initialize hamming windows
    window_a = scipy.signal.get_window("flattop", Np)
    window_g = scipy.signal.get_window("boxcar", N)

    window = window_a.reshape((-1, 1))

    window_2 = window_g
    
    window_2 = window_2/np.sum(window_2)

    conj_signal = np.conj(signal)

    # Pad signal for windowing
    signal_padded = np.pad(signal, (pad_length, pad_length))

    # Initialize memory aligned variable -> Faster fft
    Xt = pyfftw.empty_aligned((Np, N), dtype='complex')

    # Channelizer
    for i in range(Np):
        Xt[i, :] = signal_padded[i:i+N]

    # Windowing
    Xt = Xt * window

    # FFT
    Xt = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=0), axes=0)

    # Enable cache for pyfftw (faster fft)
    pyfftw.interfaces.cache.enable()
          
    # Exponential to calculate complex demodulate
    k = np.linspace(-Np/2, Np/2 - 1, Np).reshape((-1,1))
    n = np.linspace(0, N - 1, N)
    E = np.exp(-1j*2*np.pi*n*k/Np)
    
    # Multiply with original conjugated signal and second FFT
    if not conjugate:
        Xt = Xt * conj_signal * E * window_2
    else:
        Xt = Xt * signal * E * window_2

    ssca = np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(Xt, axis=1), axes=1))

    # Map spectral and cycle frequencies to SSCA output
    q = np.linspace(-N/2, N/2 - 1, N)
    f = k/(2*Np) - q/(2*N)
    alpha = k/Np + q/N

    # Optional spectral coherence calculation from SSCA SCF
    if coherence:
        sample_1 = np.rint((f + alpha/2) * Np/fs).astype(int)
        if conjugate:
            sample_2 = np.rint((alpha/2 - f) * Np/fs).astype(int)
        else:
            sample_2 = np.rint((f - alpha/2) * Np/fs).astype(int)

        smoothing_width = int(0.01 * N)

        psd_fsm = np.fft.ifftshift(psd(signal, L=Np, method="bartlett", db=False, plot=False))

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
    

def fam_flattop(signal, Np=64, L=16, conjugate=False, plot=False, coherence=False, fs=1):
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
    window_a = scipy.signal.get_window("flattop", Np)
    window_g = scipy.signal.get_window("boxcar", P)
    
    window = window_a.reshape((-1, 1))

    window = window / np.sqrt(np.sum(window**2))

    window_2 = window_g
    
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

    # if conjugate:
    #     Yt_conj = np.conj(np.conj(Xt[::-1, :]) * phase_compensation)
    #     Xt = Xt * phase_compensation
    # else:
    #     Xt = Xt * phase_compensation
    #     Yt_conj = np.conj(Xt)

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
        sample_1 = np.rint((fj + alpha/2) * Np/fs).astype(int)
        if conjugate:
            sample_2 = np.rint((alpha/2 - fj) * Np/fs).astype(int)
        else:
            sample_2 = np.rint((fj - alpha/2) * Np/fs).astype(int)

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