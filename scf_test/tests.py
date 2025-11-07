import pyfftw
from matplotlib import pyplot as plt
import numpy as np
import psk as psk
import fsk as fsk
import tracemalloc
import pytest

def add_awgn(signal, noise_power=0.1, db=False, rng=np.random.default_rng()):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal to which noise will be added
    noise_power : float, optional
        Power of the noise to be added, default=0.1
    db : bool, optional
        If True, noise_power is interpreted in decibels, default=False
        
    Returns
    -------
    noisy_signal : ndarray
        Signal with added complex AWGN
    """
    # Generate complex noise with variance/power = 1 and mean = 0
    mean = 0
    # Power are divided to real and imaginary components. If only real, variance = 1
    variance = np.sqrt(2)/2

    N = len(signal)
    noise_real = rng.normal(mean, variance, N)
    noise_imag = rng.normal(mean, variance, N)

    noise = noise_real + 1j*noise_imag

    if db:
        noise_scaled = np.sqrt(10**(noise_power/10))*noise
    else:
        noise_scaled = np.sqrt(noise_power)*noise

    return signal + noise_scaled

def add_awgn_snr(signal, desired_snr=10, db=True, rng=np.random.default_rng()):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal with specified SNR.
    
    Parameters
    ----------
    signal : array_like
        Input signal to which noise will be added
    desired_snr : float, optional
        Desired Signal-to-Noise Ratio, default=10
    db : bool, optional
        If True, desired_snr is interpreted in decibels, default=True
        
    Returns
    -------
    noisy_signal : ndarray
        Signal with added complex AWGN at the specified SNR
    """
    # Generate complex noise with variance/power = 1 and mean = 0
    mean = 0
    # Power are divided to real and imaginary components. If only real, variance = 1
    variance = np.sqrt(2)/2

    N = len(signal)
    noise_real = rng.normal(mean, variance, N)
    noise_imag = rng.normal(mean, variance, N)

    noise = noise_real + 1j*noise_imag

    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # Scale noise power to achieve desired snr
    if db:
        noise_power = signal_power/(10**(desired_snr/10))
    else:
        noise_power = signal_power/desired_snr

    noise_scaled = np.sqrt(noise_power)*noise

    return signal + noise_scaled

def add_cfo(signal, cfo):
    signal_cfo = signal * np.exp(1j*2*np.pi*cfo*np.arange(0,len(signal)))
    return signal_cfo

def plot_scf_3d(scf, cycle_frequency, fs=1, db=False, title="Spectral Correlation Density", 
              figsize=(10, 6), z_lim=None, colormap='viridis', view_angle=None):
    """
    Create a 3D wireframe plot of Spectral Correlation Function (SCF) data.
    
    Parameters
    ----------
    scf : ndarray
        Spectral correlation function data with shape (n_cycle_frequencies, n_frequencies)
    cycle_frequency : array_like
        Cycle frequencies used to generate the SCF data
    fs : float, optional
        Sampling frequency (Hz), default=1
    db : bool, optional
        If True, input SCF data is in dB and will be converted to magnitude, default=False
    title : str, optional
        Plot title, default="Spectral Correlation Density (FSM)"
    figsize : tuple, optional
        Figure size (width, height) in inches, default=(8, 6)
    z_lim : tuple, optional
        Z-axis limits as (min, max), default=None (auto-scaling)
    colormap : str, optional
        Matplotlib colormap name for the wireframe, default='viridis'
    view_angle : tuple, optional
        View angle as (elevation, azimuth), default=None (matplotlib default)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    
    Examples
    --------
    >>> plot_scf_3d(fsm_data, cycle_frequency, fs=1000, title="My SCF Plot")
    """  
    # Ensure cycle frequency is a np arrray
    cycle_frequency = np.array(cycle_frequency)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert from dB if needed
    if db:
        scf_mag = 10**(scf / 10)  # Convert from dB to magnitude
    else:
        scf_mag = np.abs(scf)  # Ensure we're plotting magnitude
    
    # Create frequency vector
    f = np.linspace(-fs/2, fs/2, scf.shape[1])
    
    # Create meshgrid for 3D plotting
    f_3d, alpha_3d = np.meshgrid(f, cycle_frequency.flatten()/fs)

    # Ensure cycle frequency is a column vector
    cycle_frequency = cycle_frequency.reshape((-1, 1))/fs

    # Create the mask to only keep principal domain region
    #    Condition for being outside the principal domain: |f/fs| + |(alpha/fs)/2| > 0.5
    mask = np.abs(f) + np.abs(cycle_frequency / 2.0) > 0.5
    
    # Apply the mask to zero out elements in scf.
    scf_mag[mask] = 0.0
    
    # Create colormap based on z values
    norm = Normalize(vmin=np.min(scf_mag), vmax=np.max(scf_mag))
    colors = cm.get_cmap(colormap)(norm(scf_mag))
    
    # Plot the wireframe
    for i in range(len(cycle_frequency)):
        ax.plot(np.ones_like(f) * cycle_frequency[i], 
                f, 
                scf_mag[i], 
                color=colors[i, 0], 
                linewidth=1.5)
    
    # Set labels and title
    ax.set_xlabel("Cycle Frequency (Hz)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("Magnitude")
    ax.set_title(title)
    
    # Set z-axis limits if provided
    if z_lim:
        ax.set_zlim(z_lim)
    
    # Set view angle if provided
    if view_angle:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    plt.tight_layout()
    
    return fig, ax

def calculate_bpsk_scf(number_of_points, alpha_slices, Tk, f_offset, conjugate=False, plot=False):
    """
    Calculates the theoretical Spectral Correlation Function (SCF) for a baseband
    BPSK signal with rectangular pulse shaping and a carrier frequency offset.

    The SCF of a signal with a frequency offset is the original SCF shifted in
    the frequency domain: S_new(f, α) = S_original(f - f_offset, α).

    Args:
        f_grid (ndarray): 2D meshgrid of frequencies.
        alpha_grid (ndarray): 2D meshgrid of cyclic frequencies.
        Tk (float): The symbol period in samples.
        f_offset (float): The carrier frequency offset.

    Returns:
        ndarray: 2D array containing the magnitude of the SCF.
    """

    f_axis = np.linspace(-1 / 2, 1 / 2, number_of_points)

    # Create a 2D meshgrid to calculate the SCF over the entire f-α plane
    f_grid, alpha_grid = np.meshgrid(f_axis, alpha_slices)

    # Apply the frequency offset to the entire frequency grid
    f_shifted_grid = f_grid - f_offset
    
    # np.sinc(x) is defined as sin(pi*x)/(pi*x)
    if conjugate:
        term1 = Tk * np.sinc((f_shifted_grid + alpha_grid/2) * Tk)
        term2 = Tk * np.sinc((alpha_grid/2 - f_grid - f_offset) * Tk)
        significant_slices = 1/Tk * np.arange(-20, 20) + 2*f_offset; 
    else:
        term1 = Tk * np.sinc((f_shifted_grid + alpha_grid/2) * Tk)
        term2 = Tk * np.sinc((f_shifted_grid - alpha_grid/2) * Tk)
        significant_slices = 1/Tk * np.arange(-20, 20);

    significant_slices = np.round(significant_slices[np.abs(significant_slices) < 1], 4)
    
    spectral_correlation = 1/Tk * np.abs(term1 * term2) # 1/Tk scaling to get accurate PSD

    # 1. Create a boolean mask. 
    #    It's True for any alpha_slice that is NOT in significant_slices.
    #    We round alpha_slices to 2 decimal places to ensure a reliable 
    #    comparison against the already-rounded significant_slices.
    is_not_significant_mask = ~np.isin(np.round(alpha_slices, 4), significant_slices)
    
    # 2. Use this mask to set the non-significant rows to zero.
    #    NumPy will select all rows where the mask is True and set their values to 0.
    spectral_correlation[is_not_significant_mask, :] = 0

    if plot:
        plot_scf_3d(spectral_correlation, alpha_slices)
        
        
    return spectral_correlation

def validation_test(func_lambda, name="algorithm", Np=512, snr=10, conjugate=False):
    rng = np.random.default_rng(11)

    start_i = 10
    end_i = 19 #17
    range_i = end_i - start_i
    
    samples_per_symbol = 10 # Symbol rate = 0.20 Hz
    cfo = 0.05

    max_no_of_symbols = 2**(end_i-1); #65536 samples
    max_length = max_no_of_symbols * samples_per_symbol; #262144 samples

    if not conjugate:
        cycle_frequency_check = 1/(2*samples_per_symbol) * np.arange(1, 20);
    else:
        cycle_frequency_check = 1/(2*samples_per_symbol) * np.arange(-20, 20);

    cycle_frequency_check = np.round(cycle_frequency_check[cycle_frequency_check < 1], 2)

    scd_ref = calculate_bpsk_scf(max_length, cycle_frequency_check, samples_per_symbol, cfo, conjugate=conjugate)

    scd_ref = np.fft.ifftshift(scd_ref, axes=1)

    scd_rmse = np.ones(range_i)
    no_of_points = np.ones(range_i)

    for i in range(start_i, end_i):
        number_of_symbols = 2**i

        # Noise first then add cfo
        bpsk_signal = create_rect_bpsk_signal(number_of_symbols, samples_per_symbol, rng=rng)
        bpsk_signal_noise = add_awgn_snr(bpsk_signal, desired_snr=snr, rng=rng)
        bpsk_signal_noise_cfo = add_cfo(bpsk_signal_noise, cfo) # Add 0.5Hz CFO
        bpsk_signal_noise_cfo = bpsk_signal_noise_cfo[:2**i]

        N = len(bpsk_signal_noise_cfo)

        print(f"Signal Length: {N}, Window Length: {Np}")
        
        spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np, conjugate)

        #Alpha binning
        alpha = np.round(alpha * N) / N

        scd_sum_square_error = 0
        num_points = 0
        fig, axs = plt.subplots(len(cycle_frequency_check), sharex=True, figsize=(10,7))
        for j in range(len(cycle_frequency_check)):
            mask = np.abs(alpha - cycle_frequency_check[j]) < 1/(2*N)
            scd_slice = spectral_corr[mask]
            f_slice = f[mask]
            f_index = np.rint(f_slice * max_length).astype(int)
            num_points += len(scd_slice)

            axs[j].plot(f_slice, scd_slice, label="Algorithm")
            axs[j].plot(f_slice, scd_ref[j, f_index], label="FSM")
            axs[j].label_outer()
            axs[j].text(0.9, 0.5, f'alpha = {cycle_frequency_check[j]}', horizontalalignment='center', verticalalignment='center', transform=axs[j].transAxes)
            
            scd_sum_square_error += np.sum((scd_slice - scd_ref[j, f_index])**2)

        handles, labels = axs[j].get_legend_handles_labels()
        axs[j].set_xlabel("Spectral Frequency (Hz)")
        fig.legend(handles, labels, loc='upper right')
        fig.suptitle(f'BPSK SCF, N={N}, Np={Np}', fontsize=12)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        #fig.savefig(f"fig/{name}_{N}_{Np}.png")

        scd_rmse[i - start_i] = np.sqrt(scd_sum_square_error/num_points)
        no_of_points[i - start_i] = np.log2(N)
        print(f"SCD RMSE: {scd_rmse[i - start_i]}")

    plt.plot(2**no_of_points, scd_rmse)
    plt.xscale('log', base=2)
    plt.xlabel("Signal Length")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Signal Length (Np={Np})")

    return scd_rmse, no_of_points

def rmse_vs_theoretical_bpsk(spectral_correlation, f, alpha, samples_per_symbol, cfo, conjugate=False):
    N = np.shape(spectral_correlation)[1]
    
    if not conjugate:
        cycle_frequency_check = 1/(2*samples_per_symbol) * np.arange(1, 20);
    else:
        cycle_frequency_check = 1/(2*samples_per_symbol) * np.arange(-21, 21) + 2*cfo;

    cycle_frequency_check = np.round(cycle_frequency_check[cycle_frequency_check < 1], 2)

    scd_ref = calculate_bpsk_scf(4*N, cycle_frequency_check, samples_per_symbol, cfo, conjugate=conjugate)

    scd_ref = np.fft.ifftshift(scd_ref, axes=1)

    #Alpha binning
    alpha = np.round(alpha * N) / N
    
    scd_sum_square_error = 0
    num_points = 0

    for j in range(len(cycle_frequency_check)):
        mask = np.abs(alpha - cycle_frequency_check[j]) < 1/(2*N)
        scd_slice = spectral_correlation[mask]
        f_slice = f[mask]
        f_index = np.rint(f_slice * 4 * N).astype(int)
        num_points += len(scd_slice)
    
        scd_sum_square_error += np.sum((scd_slice - scd_ref[j, f_index])**2)

    scd_rmse = np.sqrt(scd_sum_square_error/num_points)
    no_of_points = np.log2(N)

    return scd_rmse, no_of_points

from sklearn import metrics

def plot_roc_full(func_lambda, Np=8, no_of_simulation=1000, threshold_resolution=10000, snr=[0], N=2**15, fam=False, L=2, name="roc", save=False):
    # Create a reference CDP using no noise bpsk so we know the cycle frequency
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz
    cfo = 0.05
    srrc_filter_span = 21
    beta = 0.5
    bandwidth_time = 0.3

    average_pf_bpsk = np.zeros((len(snr), 2, threshold_resolution))
    average_pd_bpsk = np.zeros((len(snr), 2, threshold_resolution))

    average_pf_qpsk = np.zeros((len(snr), 2, threshold_resolution))
    average_pd_qpsk = np.zeros((len(snr), 2, threshold_resolution))

    average_pf_gmsk = np.zeros((len(snr), 2, threshold_resolution))
    average_pd_gmsk = np.zeros((len(snr), 2, threshold_resolution))

    roc_auc = np.zeros(len(snr))
    
    threshold = np.linspace(0, 1, threshold_resolution)
    threshold = threshold.reshape((1, -1))
    
    for i in range(len(snr)):
        signal_flag = True
        true_positive_bpsk = np.zeros((2, threshold_resolution))
        false_positive_bpsk = np.zeros((2, threshold_resolution))
        
        true_positive_qpsk = np.zeros((2, threshold_resolution))
        false_positive_qpsk = np.zeros((2, threshold_resolution))
        
        true_positive_gmsk = np.zeros((2, threshold_resolution))
        false_positive_gmsk = np.zeros((2, threshold_resolution))
        
        for j in range(no_of_simulation*2):
            # Half signal + noise, half only noise
            bpsk_signal = psk.create_rect_bpsk_signal(N, samples_per_symbol)
            bpsk_signal_cfo = add_cfo(bpsk_signal, cfo) # Add 0.5Hz CFO
            bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr[i])

            qpsk_signal = psk.create_srrc_qpsk_signal(N, samples_per_symbol, srrc_filter_span, beta)
            qpsk_signal_cfo = add_cfo(qpsk_signal, cfo) # Add 0.5Hz CFO
            qpsk_signal_noise_cfo = add_awgn_snr(qpsk_signal_cfo, desired_snr=snr[i])

            gmsk_signal, _, _ = fsk.create_gmsk_signal(N, samples_per_symbol, f_carrier=cfo, BT=bandwidth_time, gauss_filter_span=srrc_filter_span)
            gmsk_signal_noise_cfo = add_awgn_snr(gmsk_signal, desired_snr=snr[i])
            
            # Even for signal present
            if j % 2 == 0:
                test_signal_bpsk = bpsk_signal_noise_cfo[:N]
                test_signal_qpsk = qpsk_signal_noise_cfo[:N]
                test_signal_gmsk = gmsk_signal_noise_cfo[:N]
                signal_flag = True
            # Odd for noise only
            else:
                test_signal_bpsk = (bpsk_signal_noise_cfo - bpsk_signal_cfo)[:N]
                test_signal_qpsk = (qpsk_signal_noise_cfo - qpsk_signal_cfo)[:N]
                test_signal_gmsk = (gmsk_signal_noise_cfo - gmsk_signal)[:N]
                signal_flag = False

            if fam:
                spectral_coherence_bpsk_nc, f, alpha = func_lambda(test_signal_bpsk, Np=Np, L=L, conjugate=False, coherence=True)
                spectral_coherence_bpsk_c, _, _ = func_lambda(test_signal_bpsk, Np=Np, L=L, conjugate=True, coherence=True)

                spectral_coherence_qpsk_nc, _, _ = func_lambda(test_signal_qpsk, Np=Np, L=L, conjugate=False, coherence=True)
                spectral_coherence_qpsk_c, _, _ = func_lambda(test_signal_qpsk, Np=Np, L=L, conjugate=True, coherence=True)

                spectral_coherence_gmsk_nc, _, _ = func_lambda(test_signal_gmsk, Np=Np, L=L, conjugate=False, coherence=True)
                spectral_coherence_gmsk_c, _, _ = func_lambda(test_signal_gmsk, Np=Np, L=L, conjugate=True, coherence=True)
            else:
                spectral_coherence_bpsk_nc, f, alpha = func_lambda(test_signal_bpsk, L=Np, conjugate=False, coherence=True)
                spectral_coherence_bpsk_c, _, _ = func_lambda(test_signal_bpsk, L=Np, conjugate=True, coherence=True)

                spectral_coherence_qpsk_nc, _, _ = func_lambda(test_signal_qpsk, L=Np, conjugate=False, coherence=True)
                spectral_coherence_qpsk_c, _, _ = func_lambda(test_signal_qpsk, L=Np, conjugate=True, coherence=True)

                spectral_coherence_gmsk_nc, _, _ = func_lambda(test_signal_gmsk, L=Np, conjugate=False, coherence=True)
                spectral_coherence_gmsk_c, _, _ = func_lambda(test_signal_gmsk, L=Np, conjugate=True, coherence=True)

            coh_max_bpsk = []
            coh_max_qpsk = []
            coh_max_gmsk = []

            # Extract maximum coh at alpha = -0.8 for conj BPSK SCD
            mask_neg_8_c_bpsk = np.abs(alpha + 0.8) < 1/(2*N)
            coh_max_bpsk.append(np.max(spectral_coherence_bpsk_c[mask_neg_8_c_bpsk]))

            # Extract maximum coh at alpha = 0.8 for non-conj BPSK SCD
            mask_pos_8_nc_bpsk = np.abs(alpha - 0.8) < 1/(2*N)
            coh_max_bpsk.append(np.max(spectral_coherence_bpsk_nc[mask_pos_8_nc_bpsk]))

            # Extract maximum coh at alpha = 0.1 for conj QPSK SCD
            # Note: Conj QPSK is expected to be 0 
            mask_pos_1_c_qpsk = np.abs(alpha - 0.1) < 1/(2*N)
            coh_max_qpsk.append(np.max(spectral_coherence_qpsk_c[mask_pos_1_c_qpsk]))

            # Extract maximum coh at alpha = 0.1 for non-conj QPSK SCD
            mask_pos_1_nc_qpsk = np.abs(alpha - 0.1) < 1/(2*N)
            coh_max_qpsk.append(np.max(spectral_coherence_qpsk_nc[mask_pos_1_nc_qpsk]))

            # Extract maximum coh at alpha = 0.05 for conj GMSK SCD
            mask_pos_15_c_gmsk = np.abs(alpha - 0.15) < 1/(2*N)
            coh_max_gmsk.append(np.max(spectral_coherence_gmsk_c[mask_pos_15_c_gmsk]))

            # Extract maximum coh at alpha = 0.1 for non-conj GMSK SCD
            mask_pos_1_nc_gmsk = np.abs(alpha - 0.1) < 1/(2*N)
            coh_max_gmsk.append(np.max(spectral_coherence_gmsk_nc[mask_pos_1_nc_gmsk]))

            coh_max_bpsk = np.array(coh_max_bpsk).reshape((-1, 1))
            coh_max_qpsk = np.array(coh_max_qpsk).reshape((-1, 1))
            coh_max_gmsk = np.array(coh_max_gmsk).reshape((-1, 1))

            # Check if magnitude pass threshold
            cf_bpsk_detected = coh_max_bpsk >= threshold
            cf_qpsk_detected = coh_max_qpsk >= threshold
            cf_gmsk_detected = coh_max_gmsk >= threshold

            if signal_flag:
                true_positive_bpsk += cf_bpsk_detected.astype(int)
                true_positive_qpsk += cf_qpsk_detected.astype(int)
                true_positive_gmsk += cf_gmsk_detected.astype(int)
            else:
                false_positive_bpsk += cf_bpsk_detected.astype(int)
                false_positive_qpsk += cf_qpsk_detected.astype(int)
                false_positive_gmsk += cf_gmsk_detected.astype(int)

        # len(coh_max) is only relevant if you want to check other alphas
        average_pd_bpsk[i, :, :] = true_positive_bpsk/(no_of_simulation)
        average_pf_bpsk[i, :, :] = false_positive_bpsk/(no_of_simulation)

        average_pd_qpsk[i, :, :] = true_positive_qpsk/(no_of_simulation)
        average_pf_qpsk[i, :, :] = false_positive_qpsk/(no_of_simulation)

        average_pd_gmsk[i, :, :] = true_positive_gmsk/(no_of_simulation)
        average_pf_gmsk[i, :, :] = false_positive_gmsk/(no_of_simulation)
        #roc_auc[i] = metrics.auc(average_pf[i, :], average_pd[i, :])
        #plt.plot(average_pf[i, :], average_pd[i, :], label=f"{snr[i]}dB SNR, AUC={np.round(roc_auc[i], 3)}")
        
        #plt.plot(false_positive[i, :], average_pd[i, :], label=f"{snr[i]}dB SNR")

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(5, 5), sharex=True, sharey=True)
    fig.suptitle(f'ROC check, N={N}, Np={Np}', fontsize=12)
    fig.supxlabel("Probability of False Alarm")
    fig.supylabel("Probability of Detection")

    roc_auc_total = np.zeros(len(snr))

    roc_auc = np.zeros((len(snr), 2, 3))
    
    for i in range(len(snr)):
        for j in range(2):
            roc_auc_bpsk = np.round(metrics.auc(average_pf_bpsk[i, j, :], average_pd_bpsk[i, j, :]), 3)
            axs[0, j].plot(average_pf_bpsk[i, j, :], average_pd_bpsk[i, j, :], label=f"{snr[i]}dB SNR")
            axs[0, j].plot([0, 1], [0, 1], color='red', linestyle='--')
            # axs[0, j].legend()

            roc_auc_qpsk = np.round(metrics.auc(average_pf_qpsk[i, j, :], average_pd_qpsk[i, j, :]), 3)
            axs[1, j].plot(average_pf_qpsk[i, j, :], average_pd_qpsk[i, j, :], label=f"{snr[i]}dB SNR")
            axs[1, j].plot([0, 1], [0, 1], color='red', linestyle='--')
            # axs[1, j].legend()

            roc_auc_gmsk = np.round(metrics.auc(average_pf_gmsk[i, j, :], average_pd_gmsk[i, j, :]), 3)
            axs[2, j].plot(average_pf_gmsk[i, j, :], average_pd_gmsk[i, j, :], label=f"{snr[i]}dB SNR")
            axs[2, j].plot([0, 1], [0, 1], color='red', linestyle='--')
            # axs[2, j].legend()

            if j == 0:
                axs[0, j].set_title("Conjugate Rect BPSK")
                axs[1, j].set_title("Conjugate SRRC QPSK")
                axs[2, j].set_title("Conjugate GMSK")
            else:
                axs[0, j].set_title("Non-Conjugate Rect BPSK")
                axs[1, j].set_title("Non-Conjugate SRRC QPSK")
                axs[2, j].set_title("Non-Conjugate GMSK")

            roc_auc_total[i] += roc_auc_bpsk + roc_auc_qpsk + roc_auc_gmsk

            roc_auc[i, j, 0] = roc_auc_bpsk
            roc_auc[i, j, 1] = roc_auc_qpsk
            roc_auc[i, j, 2] = roc_auc_gmsk
        
        print(f"{snr[i]}dB SNR Score: {roc_auc_total[i]}/6")

        # roc_auc = np.round(metrics.auc(np.sum(average_pf[i, :, :], axis=0) / 17, np.sum(average_pd[i, :, :], axis=0) / 17), 3)
        # axs[-1].plot(np.sum(average_pf[i, :, :], axis=0) / 2, np.sum(average_pd[i, :, :], axis=0) / 17, label=f"{snr[i]}dB SNR, AUC={roc_auc}")
        # print(np.shape(average_pf[i, :, :]))
        # axs[-1].text(0.9, 0.5, "All alpha combined", horizontalalignment='center', verticalalignment='center', transform=axs[-1].transAxes)
        # axs[-1].plot([0, 1], [0, 1], color='red', linestyle='--')
        # axs[-1].legend()

    # plt.tight_layout()
    #fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize='small')
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space
    if save:
        plt.savefig("results/" + name + ".png")
    plt.show()

    average_pd = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    average_pf = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    
    return average_pd, average_pf, roc_auc, threshold_resolution

from scipy.signal.windows import get_window

def window_test(scf_func, name="algorithm_window_test", signal_length=4096, Np=64, fam=False, conjugate=False, snr=0, number_of_runs=20, plot=False):
    number_of_symbols = int(signal_length/8)
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz

    N = signal_length

    number_of_top_indices = np.rint(signal_length * 0.01).astype(int)

    print(number_of_top_indices)

    # Following Eric April's recommendation
    # a(n) or first window -> Large attenuation
    # g(n) or second window -> Small bandwidth
    #  "hamming", "hann", "cosine", "barthann", "lanczos" are middle of the road windows
    small_bandwidth_window = ["boxcar", "triang", ("kaiser", 4), "hamming", "hann", "cosine", "barthann", "lanczos", ("tukey", 0.2)]
    large_attenuation_window = [("chebwin", 100), ("kaiser", 14), "blackmanharris", "nuttall", "flattop", "parzen", "bohman", "taylor", "hamming", "hann", "cosine", "barthann", "lanczos"]

    all_windows = ["boxcar", "triang", ("kaiser", 4), "hamming", "hann", "cosine", "barthann", "lanczos", ("tukey", 0.2), ("chebwin", 100), ("kaiser", 14), "blackmanharris", "nuttall", "flattop", "parzen", "bohman", "taylor"]

    no_of_windows = len(all_windows)

    cycle_leakage_pts = np.zeros((no_of_windows, no_of_windows))

    for i in range(number_of_runs):
        bpsk_signal = psk.create_rect_bpsk_signal(number_of_symbols, samples_per_symbol)
        bpsk_signal_cfo = add_cfo(bpsk_signal, 0.05) # Add 0.5Hz CFO
        bpsk_signal_cfo_noise = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr) # Add 1W AWGN

        test_signal = bpsk_signal_cfo_noise[:signal_length]

        for i in range(len(all_windows)):
            for j in range(len(all_windows)):
                window_a = get_window(all_windows[i], Np)
                
                if fam:
                    window_j = get_window(all_windows[j], int(N/(Np/4)))
                else:
                    window_j = get_window(all_windows[j], N)
            
                scd, f, alpha = scf_func(test_signal, window_a, window_j, Np, conjugate)
            
                # top_indices = np.argpartition(scd.flatten(), -number_of_top_indices)[-number_of_top_indices:]
                # x, y = np.unravel_index(top_indices, np.shape(scd))
            
                # f_top = f[x, y]
                # alpha_top = alpha[x, y]
            
                # mask_1 = np.abs(alpha_top) > 0 + 1/(N)
            
                # mask_2 = np.abs(alpha_top) < 0.1 - 1/(N)
            
                # mask = mask_1 & mask_2
            
                # f_masked = f_top[mask]
                # alpha_masked = alpha_top[mask]

                if not conjugate:
                    mask_1 = np.abs(alpha) > 0 + 1/(N)
                    mask_2 = np.abs(alpha) < 0.1 - 1/(N)
                else:
                    mask_1 = np.abs(alpha - 0.1) > 0 + 1/(N)
                    mask_2 = np.abs(alpha - 0.1) < 0.1 - 1/(N)

                mask = mask_1 & mask_2
    
                cycle_leakage_pts[i, j] += np.sum(scd[mask])/len(scd[mask])

    cycle_leakage_pts = cycle_leakage_pts / number_of_runs

    row, col = np.unravel_index(np.argmin(cycle_leakage_pts), np.shape(cycle_leakage_pts))

    print(f"Best Windows: a(n): {all_windows[row]}, g(n): {all_windows[col]} with {cycle_leakage_pts[row, col]} average cycle leakage points.")

    if plot:
        fig = plt.figure(figsize=(5,4.5))
        pc_kwargs = {'rasterized': True, 'edgecolor': 'black'}
        plt.pcolormesh(cycle_leakage_pts, **pc_kwargs)
        plt.colorbar()
        plt.xticks(np.arange(no_of_windows) + 0.5, labels=all_windows, rotation="vertical")
        plt.yticks(np.arange(no_of_windows) + 0.5, labels=all_windows)
        plt.xlabel("g(n), Second Window")
        plt.ylabel("a(n), First Window")
        plt.title(f"Window Test, N={signal_length}, Np={Np}, SNR={snr}dB")
        plt.tight_layout()
        plt.savefig("results/" + name + ".png")
        plt.show()

    return cycle_leakage_pts

def rmse_window_test(scf_func, signal_length=4096, Np=64, fam=False, snr=0, conjugate=False, plot=False, rng=np.random.default_rng()):
    number_of_symbols = signal_length
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz
    cfo = 0.05

    bpsk_signal = psk.create_rect_bpsk_signal(number_of_symbols, samples_per_symbol, rng=rng)
    bpsk_signal_noise = add_awgn_snr(bpsk_signal, desired_snr=snr) # Add 1W AWGN
    bpsk_signal_noise_cfo = add_cfo(bpsk_signal_noise, cfo) # Add 0.5Hz CFO

    if plot:
        plot_signal(bpsk_signal_noise_cfo, title="BPSK with noise and Carrier Offset")

    test_signal = bpsk_signal_noise_cfo[:signal_length]

    N = len(test_signal)

    # Following Eric April's recommendation
    # a(n) or first window -> Large attenuation
    # g(n) or second window -> Small bandwidth
    #  "hamming", "hann", "cosine", "barthann", "lanczos" are middle of the road windows
    small_bandwidth_window = ["boxcar", "triang", ("kaiser", 4), "hamming", "hann", "cosine", "barthann", "lanczos", ("tukey", 0.2)]
    large_attenuation_window = [("chebwin", 100), ("kaiser", 14), "blackmanharris", "nuttall", "flattop", "parzen", "bohman", "taylor", "hamming", "hann", "cosine", "barthann", "lanczos"]

    all_windows = ["boxcar", "triang", ("kaiser", 4), "hamming", "hann", "cosine", "barthann", "lanczos", ("tukey", 0.2), ("chebwin", 100), ("kaiser", 14), "blackmanharris", "nuttall", "flattop", "parzen", "bohman", "taylor"]

    no_of_windows = len(all_windows)

    cycle_leakage_pts = np.zeros((no_of_windows, no_of_windows))
    rmse = np.zeros((no_of_windows, no_of_windows))

    bpsk_signal = psk.create_rect_bpsk_signal(number_of_symbols, samples_per_symbol, rng=rng)
    bpsk_signal_noise = add_awgn_snr(bpsk_signal, desired_snr=snr) # Add 1W AWGN
    bpsk_signal_noise_cfo = add_cfo(bpsk_signal_noise, cfo) # Add 0.5Hz CFO
    for i in range(len(all_windows)):
        for j in range(len(all_windows)):
            window_a = get_window(all_windows[i], Np)
            
            if fam:
                window_j = get_window(all_windows[j], int(N/(Np/4)))
            else:
                window_j = get_window(all_windows[j], N)
        
            scd, f, alpha = scf_func(test_signal, window_a, window_j, Np, conjugate)

            rmse[i, j], length = rmse_vs_theoretical_bpsk(scd, f, alpha, samples_per_symbol, cfo, conjugate=conjugate)

            if not conjugate:
                mask_1 = np.abs(alpha) > 0 + 1/N
                mask_2 = np.abs(alpha) < 0.1 - 1/N
            else:
                mask_1 = np.abs(alpha - 0.1) > 0 + 1/N
                mask_2 = np.abs(alpha - 0.1) < 0.1 - 1/N
        
            mask = mask_1 & mask_2

            cycle_leakage_pts[i, j] = np.sum(scd[mask])/len(scd[mask].flatten())

    if plot:
        pc_kwargs = {'rasterized': True, 'edgecolor': 'black'}

        fig, axes = plt.subplots(3, 1, figsize=(7,20))
        
        pos = axes[0].pcolormesh(rmse, **pc_kwargs)
        fig.colorbar(pos, ax=axes[0])
        axes[0].set_xticks(np.arange(no_of_windows) + 0.5, labels=all_windows, rotation="vertical")
        axes[0].set_yticks(np.arange(no_of_windows) + 0.5, labels=all_windows)
        axes[0].set_xlabel("g(n), Second Window")
        axes[0].set_ylabel("a(n), First Window")
        axes[0].set_title(f"RMSE Window Test, N={signal_length}, Np={Np}")

        pos = axes[1].pcolormesh(cycle_leakage_pts, **pc_kwargs)
        fig.colorbar(pos, ax=axes[1])
        axes[1].set_xticks(np.arange(no_of_windows) + 0.5, labels=all_windows, rotation="vertical")
        axes[1].set_yticks(np.arange(no_of_windows) + 0.5, labels=all_windows)
        axes[1].set_xlabel("g(n), Second Window")
        axes[1].set_ylabel("a(n), First Window")
        axes[1].set_title(f"Cycle Leakage % Window Test, N={signal_length}, Np={Np}")

        combined_error = cycle_leakage_pts + rmse
        pos = axes[2].pcolormesh(combined_error, **pc_kwargs)
        fig.colorbar(pos, ax=axes[2])
        axes[2].set_xticks(np.arange(no_of_windows) + 0.5, labels=all_windows, rotation="vertical")
        axes[2].set_yticks(np.arange(no_of_windows) + 0.5, labels=all_windows)
        axes[2].set_xlabel("g(n), Second Window")
        axes[2].set_ylabel("a(n), First Window")
        axes[2].set_title(f"Combined Error (cycle leakage points x rmse), N={signal_length}, Np={Np}")
        fig.tight_layout()
        plt.show()

        row, col = np.unravel_index(np.argmin(rmse), np.shape(rmse))
        print(f"Best Window for RMSE: (a(n), {all_windows[row]}), (g(n), {all_windows[col]}) with {np.round(rmse[row, col], 2)} RMSE") 

        row, col = np.unravel_index(np.argmin(cycle_leakage_pts), np.shape(rmse))
        print(f"Best Window for Cycle Leakage Points: (a(n), {all_windows[row]}), (g(n), {all_windows[col]}) with {cycle_leakage_pts[row, col]}%") 

        row, col = np.unravel_index(np.argmin(combined_error), np.shape(rmse))
        print(f"Best Window Combined Error: (a(n), {all_windows[row]}), (g(n), {all_windows[col]}) with {np.round(combined_error[row, col],2)} combined error") 
        

    return cycle_leakage_pts, rmse

def window_test(scf_func, signal_length=4096, Np=64, conjugate=False, snr=0, number_of_runs=20):
    number_of_symbols = int(signal_length/8)
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz

    N = signal_length

    average_cycle_leakage = 0

    for i in range(number_of_runs):
        bpsk_signal = pskcreate_rect_bpsk_signal(number_of_symbols, samples_per_symbol)
        bpsk_signal_cfo = add_cfo(bpsk_signal, 0.05) # Add 0.5Hz CFO
        bpsk_signal_cfo_noise = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr) # Add 1W AWGN

        test_signal = bpsk_signal_cfo_noise[:signal_length]
            
        scd, f, alpha = scf_func(test_signal, Np, conjugate)

        if not conjugate:
            mask_1 = np.abs(alpha) > 0 + 1/(N)
            mask_2 = np.abs(alpha) < 0.1 - 1/(N)
        else:
            mask_1 = np.abs(alpha - 0.1) > 0 + 1/(N)
            mask_2 = np.abs(alpha - 0.1) < 0.1 - 1/(N)

        mask = mask_1 & mask_2

        average_cycle_leakage += np.sum(scd[mask])/len(scd[mask])

    average_cycle_leakage = average_cycle_leakage / number_of_runs

    return average_cycle_leakage

def speed_test(python_file_name, plot=True):
    !pytest python_file_name --benchmark-only --benchmark-save=run_np --benchmark-save-data --benchmark-disable-gc

    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    import re # Import the regular expression module
    
    # Find and load the latest benchmark JSON file
    try:
        latest_file = max(glob.glob('.benchmarks/*/*run_np*.json'))
        print(f"Loading data from: {latest_file}")
        with open(latest_file, 'r') as f:
            data = json.load(f)
    except (ValueError, FileNotFoundError):
        print("Benchmark file not found. Please run your pytest benchmark first.")
        data = None
    
    if data:
        # Convert to a DataFrame and expand the 'stats' column
        df = pd.DataFrame(data['benchmarks'])
        stats_df = df['stats'].apply(pd.Series)
        df = pd.concat([df[['name', 'group']], stats_df], axis=1)
    
        # --- Data Processing for Line Plot ---
        
        # Function to extract algorithm and parameter size from the name
        def parse_name(name):
            # This regex looks for text between brackets, e.g., test_ssca[1000]
            match = re.search(r'\[(.*)\]', name)
            if not match:
                return None, None
            
            param = int(match.group(1)) # The parameter value (e.g., 1000)
            
            # Identify the algorithm
            if 'ssca' in name:
                algo = 'ssca'
            elif 'fam' in name:
                algo = 'fam'
            else:
                algo = 'unknown'
                
            return algo, param
    
        # Apply the function to create new columns
        df[['algorithm', 'np']] = df['name'].apply(parse_name).apply(pd.Series)
        
        # Drop rows that couldn't be parsed and sort the data
        df.dropna(subset=['algorithm', 'np'], inplace=True)
        df.sort_values(by='np', inplace=True)
    
        # Display the processed DataFrame
        if plot:
            print("\nProcessed DataFrame:")
            print(df[['algorithm', 'np', 'mean', 'stddev']].head())

    # Ensure the DataFrame 'df' from the previous step exists
    if 'df' in locals() and not df.empty and plot:
        plt.figure(figsize=(10, 6))
        
        # Plot a line for each algorithm
        for algo_name in df['algorithm'].unique():
            # Select the data for the current algorithm
            subset = df[df['algorithm'] == algo_name]
            
            plt.errorbar(subset['np'], subset['mean'], marker='o', linestyle='-', label=algo_name, yerr=subset['stddev'], capsize=4)
    
        # Add labels and title
        plt.xlabel('Np size')
        plt.ylabel('Mean Execution Time (seconds)')
        plt.title('SSCA vs. FAM Performance Comparison (N=32768)')
        
        # Optional: Use a logarithmic scale if times vary widely
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        
        plt.grid(True, which="both", ls="--")
        plt.legend() # Show the legend with algorithm names
        plt.tight_layout()
        plt.show()

def memory_test(func_lambda, name="algorithm", Np=8, max_log_2=18, no_of_run=20, plot=True):
    start_i = 10
    end_i = max_log_2 #17
    range_i = end_i - start_i

    rng = np.random.default_rng()
    signal_length = 2**(np.arange(start_i, end_i))

    average_peak_usage = np.zeros(end_i - start_i)
    standard_deviation = np.zeros(end_i - start_i)

    for i in range(start_i, end_i):
        N = 2**i
        signal = rng.uniform(-1, 1, N) + rng.uniform(-1, 1, N) * 1j
        # print(f"Signal Length: {N}, Window Length: {Np}")
        peak_usages = []
        
        for j in range(no_of_run):
            tracemalloc.start() # Start tracing for this specific run

            # Execute the function
            _, _, _ =  func_lambda(signal, Np=Np, conjugate=False)
        
            # Get the peak memory for this run
            _, peak = tracemalloc.get_traced_memory()
            peak_usages.append(peak)
        
            tracemalloc.stop() # Stop and clean up the trace for this run

        peak_usages = np.round(np.array(peak_usages)/ 1024**2, 2)
        average_peak_usage[i - start_i] = sum(peak_usages) / len(peak_usages)
        standard_deviation[i - start_i] = np.std(peak_usages)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.errorbar(signal_length, average_peak_usage, yerr=standard_deviation, marker='o', capsize=4)
        plt.title(f'{name} Memory Usage vs. Signal Length (Np = 8)')
        plt.xlabel('Signal Length (N)')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.xscale('log', base=2)


    return average_peak_usage, standard_deviation, signal_length