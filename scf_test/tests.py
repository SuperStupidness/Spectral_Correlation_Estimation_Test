from matplotlib import pyplot as plt
import numpy as np
from .psk import create_srrc_qpsk_signal, create_rect_bpsk_signal
from .fsk import create_gmsk_signal
from .visualization.time_domain import plot_signal
import tracemalloc
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re # Import the regular expression module
import time
import os

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
        
    return spectral_correlation

def validation_test(func_lambda, name="algorithm", Np=512, snr=10, max_log_2=20, no_of_run=1, alpha_max = 1.0, conjugate=False, plot=True, save=True, fam=False):
    start_i = 10 # or 1024 (change if you want higher Np)
    end_i = max_log_2 
    range_i = end_i - start_i
    
    samples_per_symbol = 10 # Symbol rate = 0.20 Hz
    cfo = 0.05

    max_no_of_symbols = 2**(end_i-1); #65536 samples
    max_length = max_no_of_symbols * samples_per_symbol; #262144 samples

    if conjugate:
        cycle_frequency_check = 1/(samples_per_symbol) * np.arange(-30, 30) + 2*cfo;
    else:
        cycle_frequency_check = 1/(samples_per_symbol) * np.arange(1, 30);
    
    cycle_frequency_check = np.round(cycle_frequency_check[np.abs(cycle_frequency_check) < alpha_max], 2)

    scd_ref = calculate_bpsk_scf(max_length, cycle_frequency_check, samples_per_symbol, cfo, conjugate=conjugate)

    # print(np.shape(scd_ref))

    scd_ref = np.fft.ifftshift(scd_ref, axes=1)

    scd_rmse = np.ones(range_i)
    signal_length = 2**(np.arange(start_i, end_i))

    for i in range(start_i, end_i):
        scd_sum_square_error = 0
        num_points = 0
        number_of_symbols = 2**i
        N = 2**i 
        # print(f"Signal Length: {N}, Window Length: {Np}")

        if plot : 
            fig, axs = plt.subplots(len(cycle_frequency_check), sharex=True, figsize=(10,9))
            fig.suptitle(f'BPSK SCF, N={N}, Np={Np}', fontsize=12)
            plt.subplots_adjust(hspace=0.5)
        
        for j in range(no_of_run):
            rng=np.random.default_rng()
            # Noise first then add cfo
            bpsk_signal = create_rect_bpsk_signal(number_of_symbols, samples_per_symbol, rng=rng)
            bpsk_signal_cfo = add_cfo(bpsk_signal, cfo) # Add 0.5Hz CFO
            bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr, rng=rng)
            bpsk_signal_noise_cfo = bpsk_signal_noise_cfo[:N]

            if fam:
                spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, L=Np/4, conjugate=conjugate)
            else:
                spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, conjugate=conjugate)

            # Remove any value outside of the principal diamond
            mask_principal = np.abs(f) + np.abs(alpha / 2.0) < 0.5

            # alpha binning. Assuming cycle frequency resolution is N
            alpha = np.round(alpha * N) / N

            for k in range(len(cycle_frequency_check)):
                mask = np.abs(alpha - cycle_frequency_check[k]) <= 1/(2*N)
                scd_slice = spectral_corr[mask & mask_principal]
                f_slice = f[mask & mask_principal]
                f_index = np.rint(f_slice * max_length).astype(int)
                num_points += len(scd_slice)
                scd_sum_square_error += np.sum((scd_slice - scd_ref[k, f_index])**2)

                if plot and j == no_of_run - 1:
                    axs[k].plot(f_slice, scd_slice, label=name)
                    axs[k].plot(f_slice, scd_ref[k, f_index], label="Theory", color="orange")
                    axs[k].label_outer()
                    axs[k].text(0.9, 0.5, f'alpha = {cycle_frequency_check[k]}', horizontalalignment='center', verticalalignment='center', transform=axs[k].transAxes)
                elif plot:
                    axs[k].plot(f_slice, scd_slice)
                
        if plot:
            handles, labels = axs[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            axs[-1].set_xlabel("Spectral Frequency (Hz)")
            plt.show()

        scd_rmse[i - start_i] = np.sqrt(scd_sum_square_error/num_points)
        # print(f"SCD RMSE: {scd_rmse[i - start_i]}")

    if plot: 
        plt.plot(signal_length, scd_rmse)
        plt.xscale('log', base=2)
        plt.xlabel("Signal Length (log2 scale)")
        plt.ylabel("RMSE")
        plt.title(f"RMSE vs Signal Length (Np={Np})")
        if save:
            plt.savefig(f"fig/{name}_validation.png")

    return scd_rmse, signal_length

def extended_validation_test(func_lambda, name="algorithm", np_arr=2**np.arange(3, 10), snr_arr=np.arange(11), max_log_2=20, no_of_run=5, fam=False, conjugate=False, plot=True, save=True):
    snr_size = len(snr_arr)
    rmse_size = int(max_log_2 - 1 - np.log2(np_arr[-1]))
    np_size = len(np_arr)

    # print(rmse_size)
    
    rmse_mat = np.zeros((snr_size, np_size, rmse_size))
    signal_length_mat = np.zeros((snr_size, np_size, rmse_size))

    
    for i in range(snr_size):
        for j in range(np_size):
            print(f"SNR = {snr_arr[i]}, Np = {np_arr[j]}")
            rmse_mat[i, j, :], signal_length_mat[i, j, :] = validation_test(func_lambda, Np=np_arr[j], snr=snr_arr[i], conjugate=conjugate, 
                                                                            max_log_2=max_log_2, no_of_run=no_of_run, fam=fam, plot=False, save=False)
    
    if plot:
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(8, 6))

        # Find global min/max for the color scale
        vmin = np.min(rmse_mat)
        vmax = np.max(rmse_mat)

        # Initial plot setup (for the first SNR value)
        im = ax.pcolormesh(
            2**np.log2(np_arr), signal_length_mat[0, 0, :], rmse_mat[0, :, :].T,
            shading='gouraud', cmap='viridis', vmin=0.0, vmax=1.0
        )

        # Setup axes and colorbar
        ax.set_yscale('log', base=2)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Window Size')
        ax.set_ylabel('Signal Length')
        fig.colorbar(im, label='RMSE')

        min_marker, = ax.plot([], [], 'x', mfc='none', mec='red', mew=2, markersize=12)

        # Animation update function
        def update(frame):
            # Get the data for the current frame (SNR level)
            data = rmse_mat[frame, :, :].T
            # Update the plot data
            im.set_array(data.ravel())

            #--- NEW: Find min and update the marker's position ---
            min_flat_idx = np.argmin(rmse_mat[frame, :, :])
            min_mesh_idx = np.unravel_index(min_flat_idx, rmse_mat[frame, :, :].shape)
            min_window_size = (2**np.log2(np_arr))[min_mesh_idx[0]]
            min_signal_length = signal_length_mat[0, 0, min_mesh_idx[1]]
            
            min_marker.set_data([min_window_size], [min_signal_length]) # Pass coordinates as lists

            min_rmse = np.round(np.min(rmse_mat[frame, :, :]), decimals=5)

            # Update the title
            ax.set_title(f'{name} RMSE Heatmap, SNR = {frame}dB, min={min_rmse}')
            return im,

        # Create and save the animation
        ani = animation.FuncAnimation(fig, update, frames=11, interval=750, blit=True)

        if save:
            ani.save(f'{name}_rmse_animation.gif', writer='Pillow') # Or 'ffmpeg' for mp4

        plt.show()

    return rmse_mat, signal_length_mat

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

def plot_roc_full(func_lambda, Np=8, no_of_simulation=1000, threshold_resolution=10000, snr=[0], N=2**12, fam=False, L=2, name="roc", save=False):
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
            bpsk_signal = create_rect_bpsk_signal(N, samples_per_symbol)
            bpsk_signal_cfo = add_cfo(bpsk_signal, cfo) # Add 0.5Hz CFO
            bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr[i])

            qpsk_signal = create_srrc_qpsk_signal(N, samples_per_symbol, srrc_filter_span, beta)
            qpsk_signal_cfo = add_cfo(qpsk_signal, cfo) # Add 0.5Hz CFO
            qpsk_signal_noise_cfo = add_awgn_snr(qpsk_signal_cfo, desired_snr=snr[i])

            gmsk_signal, _, _ = create_gmsk_signal(N, samples_per_symbol, f_carrier=cfo, BT=bandwidth_time, gauss_filter_span=srrc_filter_span)
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
                spectral_coherence_bpsk_nc, f, alpha = func_lambda(test_signal_bpsk, Np=Np, conjugate=False, coherence=True)
                spectral_coherence_bpsk_c, _, _ = func_lambda(test_signal_bpsk, Np=Np, conjugate=True, coherence=True)

                spectral_coherence_qpsk_nc, _, _ = func_lambda(test_signal_qpsk, Np=Np, conjugate=False, coherence=True)
                spectral_coherence_qpsk_c, _, _ = func_lambda(test_signal_qpsk, Np=Np, conjugate=True, coherence=True)

                spectral_coherence_gmsk_nc, _, _ = func_lambda(test_signal_gmsk, Np=Np, conjugate=False, coherence=True)
                spectral_coherence_gmsk_c, _, _ = func_lambda(test_signal_gmsk, Np=Np, conjugate=True, coherence=True)

            coh_max_bpsk = []
            coh_max_qpsk = []
            coh_max_gmsk = []

            # alpha binning. Assuming cycle frequency resolution is N
            alpha = np.round(alpha * N) / N

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

            # Extract maximum coh at alpha = 0.15 for conj GMSK SCD
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

    # plt.tight_layout()
    #fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize='small')
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space
    if save:
        plt.savefig("results/" + name + ".png")
    plt.show()

    average_pd = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    average_pf = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    
    return average_pd, average_pf, roc_auc, threshold_resolution


def plot_roc_limited(func_lambda, Np=8, no_of_simulation=1000, threshold_resolution=10000, snr=[0], N=2**12, fam=False, L=2, name="roc", save=False):
    # Create a reference CDP using no noise bpsk so we know the cycle frequency
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz
    cfo = 0.05
    srrc_filter_span = 21
    beta = 0.25
    bandwidth_time = 0.3
        

    average_pf_bpsk = np.zeros((len(snr), threshold_resolution))
    average_pd_bpsk = np.zeros((len(snr), threshold_resolution))

    average_pf_qpsk = np.zeros((len(snr), threshold_resolution))
    average_pd_qpsk = np.zeros((len(snr), threshold_resolution))

    average_pf_gmsk = np.zeros((len(snr), threshold_resolution))
    average_pd_gmsk = np.zeros((len(snr), threshold_resolution))

    roc_auc = np.zeros(len(snr))
    
    threshold = np.linspace(0, 1, threshold_resolution)
    threshold = threshold.reshape((1, -1))
    
    for i in range(len(snr)):
        signal_flag = True
        true_positive_bpsk = np.zeros((1, threshold_resolution))
        false_positive_bpsk = np.zeros((1, threshold_resolution))
        
        true_positive_qpsk = np.zeros((1, threshold_resolution))
        false_positive_qpsk = np.zeros((1, threshold_resolution))
        
        true_positive_gmsk = np.zeros((1, threshold_resolution))
        false_positive_gmsk = np.zeros((1, threshold_resolution))
        
        for j in range(no_of_simulation*2):
            # Half signal + noise, half only noise
            bpsk_signal = create_rect_bpsk_signal(N, samples_per_symbol)
            bpsk_signal_cfo = add_cfo(bpsk_signal, cfo) # Add 0.5Hz CFO
            bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr[i])

            qpsk_signal = create_srrc_qpsk_signal(N, samples_per_symbol, srrc_filter_span, beta)
            qpsk_signal_cfo = add_cfo(qpsk_signal, cfo) # Add 0.5Hz CFO
            qpsk_signal_noise_cfo = add_awgn_snr(qpsk_signal_cfo, desired_snr=snr[i])

            gmsk_signal, _, _ = create_gmsk_signal(N, samples_per_symbol, f_carrier=cfo, BT=bandwidth_time, gauss_filter_span=srrc_filter_span)
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
                spectral_coherence_bpsk_nc, _, alpha = func_lambda(test_signal_bpsk, Np=Np, L=L, conjugate=False, coherence=True)

                spectral_coherence_qpsk_nc, _, _ = func_lambda(test_signal_qpsk, Np=Np, L=L, conjugate=False, coherence=True)

                spectral_coherence_gmsk_nc, _, _ = func_lambda(test_signal_gmsk, Np=Np, L=L, conjugate=False, coherence=True)
            else:
                spectral_coherence_bpsk_nc, _, alpha = func_lambda(test_signal_bpsk, Np=Np, conjugate=False, coherence=True)

                spectral_coherence_qpsk_nc, _, _ = func_lambda(test_signal_qpsk, Np=Np, conjugate=False, coherence=True)

                spectral_coherence_gmsk_nc, _, _ = func_lambda(test_signal_gmsk, Np=Np, conjugate=False, coherence=True)
        
            coh_max_bpsk = []
            coh_max_qpsk = []
            coh_max_gmsk = []

            # alpha binning. Assuming cycle frequency resolution is N
            alpha = np.round(alpha * N) / N
            
            # Extract maximum coh at alpha = 0.4 for non-conj BPSK SCD
            mask_pos_8_nc_bpsk = np.abs(alpha - 0.4) <= 1/(2*N)
            coh_max_bpsk.append(np.max(spectral_coherence_bpsk_nc[mask_pos_8_nc_bpsk]))

            # Extract maximum coh at alpha = 0.1 for non-conj QPSK SCD
            mask_pos_1_nc_qpsk = np.abs(alpha - 0.1) <= 1/(2*N)
            coh_max_qpsk.append(np.max(spectral_coherence_qpsk_nc[mask_pos_1_nc_qpsk]))

            # Extract maximum coh at alpha = 0.1 for non-conj GMSK SCD
            mask_pos_1_nc_gmsk = np.abs(alpha - 0.1) <= 1/(2*N)
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
        average_pd_bpsk[i, :] = true_positive_bpsk/(no_of_simulation)
        average_pf_bpsk[i, :] = false_positive_bpsk/(no_of_simulation)

        average_pd_qpsk[i, :] = true_positive_qpsk/(no_of_simulation)
        average_pf_qpsk[i, :] = false_positive_qpsk/(no_of_simulation)

        average_pd_gmsk[i, :] = true_positive_gmsk/(no_of_simulation)
        average_pf_gmsk[i, :] = false_positive_gmsk/(no_of_simulation)
        #roc_auc[i] = metrics.auc(average_pf[i, :], average_pd[i, :])
        #plt.plot(average_pf[i, :], average_pd[i, :], label=f"{snr[i]}dB SNR, AUC={np.round(roc_auc[i], 3)}")
        
        #plt.plot(false_positive[i, :], average_pd[i, :], label=f"{snr[i]}dB SNR")

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 5), sharex=True, sharey=True)
    fig.suptitle(f'ROC check, N={N}, Np={Np}', fontsize=12)
    fig.supxlabel("Probability of False Alarm")
    fig.supylabel("Probability of Detection")

    roc_auc_total = np.zeros(len(snr))

    roc_auc = np.zeros((len(snr), 3))
    
    for i in range(len(snr)):
        roc_auc_bpsk = np.round(metrics.auc(average_pf_bpsk[i, :], average_pd_bpsk[i, :]), 3)
        axs[0].plot(average_pf_bpsk[i, :], average_pd_bpsk[i, :], label=f"{snr[i]}dB SNR")
        axs[0].plot([0, 1], [0, 1], color='red', linestyle='--')
        axs[0].legend()

        roc_auc_qpsk = np.round(metrics.auc(average_pf_qpsk[i, :], average_pd_qpsk[i, :]), 3)
        axs[1].plot(average_pf_qpsk[i, :], average_pd_qpsk[i, :], label=f"{snr[i]}dB SNR")
        axs[1].plot([0, 1], [0, 1], color='red', linestyle='--')
        # axs[1, j].legend()

        roc_auc_gmsk = np.round(metrics.auc(average_pf_gmsk[i, :], average_pd_gmsk[i, :]), 3)
        axs[2].plot(average_pf_gmsk[i, :], average_pd_gmsk[i, :], label=f"{snr[i]}dB SNR")
        axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')
        # axs[2, j].legend()

        axs[0].set_title("Non-Conjugate Rect BPSK")
        axs[1].set_title("Non-Conjugate SRRC QPSK")
        axs[2].set_title("Non-Conjugate GMSK")

        roc_auc_total[i] += roc_auc_bpsk + roc_auc_qpsk + roc_auc_gmsk

        roc_auc[i, 0] = roc_auc_bpsk
        roc_auc[i, 1] = roc_auc_qpsk
        roc_auc[i, 2] = roc_auc_gmsk
        
        print(f"{snr[i]}dB SNR Score: {roc_auc_total[i]}/3")

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space
    if save:
        plt.savefig("fig/" + name + "_roc_limited" + ".png")
    plt.show()

    average_pd = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    average_pf = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    
    return average_pd, average_pf, roc_auc, threshold_resolution

from scipy.signal.windows import get_window

def extended_window_test(scf_func, name="algorithm_window_test", signal_length=4096, Np=64, L=16, fam=False, conjugate=False, snr=0, number_of_runs=100, plot=False):
    number_of_symbols = int(signal_length/8)
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz

    N = signal_length

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
        bpsk_signal = create_rect_bpsk_signal(number_of_symbols, samples_per_symbol)
        bpsk_signal_cfo = add_cfo(bpsk_signal, 0.05) # Add 0.5Hz CFO
        bpsk_signal_cfo_noise = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr) # Add 1W AWGN

        test_signal = bpsk_signal_cfo_noise[:signal_length]

        for i in range(len(all_windows)):
            for j in range(len(all_windows)):
                window_a = get_window(all_windows[i], Np)
                
                if fam:
                    window_j = get_window(all_windows[j], int(N/L))
                    scd, _, alpha = scf_func(test_signal, window_a, window_j, Np, L, conjugate)
                else:
                    window_j = get_window(all_windows[j], N)
                    scd, _, alpha = scf_func(test_signal, window_a, window_j, Np, conjugate)
            
                

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

    print(f"Best Windows: a(n): {all_windows[row]}, g(n): {all_windows[col]} with {cycle_leakage_pts[row, col]} average cycle leakage magnitudes.")

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
        plt.savefig("fig/" + name + ".png")
        plt.show()

    return cycle_leakage_pts

def window_test(scf_func, name="algorithm", signal_length=4096, Np=64, L=16,  conjugate=False, snr=[0], number_of_runs=20, fam=False, plot=True):
    number_of_symbols = int(signal_length/8)
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz

    N = signal_length

    average_cycle_leakage = np.zeros(len(snr))

    for i in range(len(snr)):
        for j in range(number_of_runs):
            bpsk_signal = create_rect_bpsk_signal(number_of_symbols, samples_per_symbol)
            bpsk_signal_cfo = add_cfo(bpsk_signal, 0.05) # Add 0.5Hz CFO
            bpsk_signal_cfo_noise = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr[i])

            test_signal = bpsk_signal_cfo_noise[:signal_length]
            
            if fam:
                scd, _, alpha = scf_func(test_signal, Np=Np, L=L, conjugate=conjugate)
            else:
                scd, _, alpha = scf_func(test_signal, Np=Np, conjugate=conjugate)

            if not conjugate:
                mask_1 = np.abs(alpha) > 0 + 1/(N)
                mask_2 = np.abs(alpha) < 0.1 - 1/(N)
            else:
                mask_1 = np.abs(alpha - 0.1) > 0 + 1/(N)
                mask_2 = np.abs(alpha - 0.1) < 0.1 - 1/(N)

            mask = mask_1 & mask_2

            average_cycle_leakage[i] += np.sum(scd[mask])/len(scd[mask])

        average_cycle_leakage[i] = average_cycle_leakage[i] / number_of_runs

    if plot:
        fig = plt.figure(figsize=(5, 3))
        plt.plot(snr, average_cycle_leakage, 'x--')
        plt.title(f"{name} Cycle Leakage Test: average cycle leakage magnitude {np.mean(average_cycle_leakage): .3f}")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Magnitude")
        plt.show()

    return average_cycle_leakage

def run_benchmark(benchmark_filename, algorithm_name=['ssca', 'fam'], param_name='param'):
    """
    Run pytest benchmark and return processed DataFrame.
    
    Parameters
    ----------
    benchmark_filename : str
        Path to the pytest file
    algorithm_name : list
        List of algorithm names to look for in test names
    param_name : str
        Name to give the parameter column (for clarity in DataFrame)
    
    Returns
    -------
    pd.DataFrame
        Processed benchmark results
    """
    timestamp = int(time.time())
    save_name = f'speed_test_{timestamp}'
    
    result = subprocess.run(
        ['python', '-m', 'pytest', benchmark_filename, '--benchmark-only', 
         f'--benchmark-save={save_name}', '--benchmark-save-data', 
         '--benchmark-disable-gc', '--benchmark-min-rounds=50'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("Pytest failed!")
        return None

    # Find the benchmark file
    try:
        matching_files = glob.glob(f'.benchmarks/**/*{save_name}*.json', recursive=True)
        if not matching_files:
            matching_files = glob.glob('.benchmarks/**/*.json', recursive=True)
        
        if matching_files:
            latest_file = max(matching_files, key=os.path.getmtime)
            print(f"Loading data from: {latest_file}")
            with open(latest_file, 'r') as f:
                data = json.load(f)
        else:
            print("No benchmark files found!")
            return None
    except Exception as e:
        print(f"Error loading benchmark file: {e}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(data['benchmarks'])
    stats_df = df['stats'].apply(pd.Series)
    df = pd.concat([df[['name', 'group']], stats_df], axis=1)

    # Parse test names
    def parse_name(name):
        match = re.search(r'\[(.*)\]', name)
        if not match:
            return None, None
        
        # Try to convert param to number, keep as string if not possible
        param_str = match.group(1)
        try:
            param = float(param_str) if '.' in param_str else int(param_str)
        except ValueError:
            param = param_str
        
        # Identify the algorithm
        for algo in algorithm_name:
            if algo in name:
                return algo, param
        
        return 'unknown', param

    df[['algorithm', param_name]] = df['name'].apply(parse_name).apply(pd.Series)
    df.dropna(subset=['algorithm', param_name], inplace=True)
    df.sort_values(by=param_name, inplace=True)
    
    return df


def plot_benchmark(df, param_name='param', title=None, xlabel=None, 
                   log_x=True, log_y=True, figsize=(6, 4), save_path=None):
    """
    Plot benchmark results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from run_benchmark()
    param_name : str
        Column name for x-axis parameter
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label (defaults to param_name)
    log_x, log_y : bool
        Whether to use log scale
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if df is None or df.empty:
        print("No data to plot!")
        return
    
    plt.figure(figsize=figsize)
    
    for algo_name in df['algorithm'].unique():
        subset = df[df['algorithm'] == algo_name]
        plt.errorbar(subset[param_name], subset['mean'], 
                     marker='o', linestyle='-', label=algo_name, 
                     yerr=subset['stddev'], capsize=4)

    plt.xlabel(xlabel or param_name)
    plt.ylabel('Mean Execution Time (seconds)')
    plt.title(title or 'Benchmark Comparison')
    
    if log_x:
        plt.xscale('log', base=2)
    if log_y:
        plt.yscale('log', base=2)
    
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def memory_test(func_lambda, name="algorithm", Np=8, L=2, max_log_2=18, no_of_run=10, fam=True, plot=True):
    start_i = 10
    end_i = max_log_2 #18

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
            if fam:
                _, _, _ = func_lambda(signal, Np=Np, L=L, conjugate=False)
            else:
                _, _, _ =  func_lambda(signal, Np=Np, conjugate=False)
        
            # Get the peak memory for this run
            _, peak = tracemalloc.get_traced_memory()
            peak_usages.append(peak)
        
            tracemalloc.stop() # Stop and clean up the trace for this run

        peak_usages = np.round(np.array(peak_usages)/ 1024**2, 2)
        average_peak_usage[i - start_i] = sum(peak_usages) / len(peak_usages)
        standard_deviation[i - start_i] = np.std(peak_usages)

    if plot:
        plt.figure(figsize=(5, 3))
        plt.errorbar(signal_length, average_peak_usage, yerr=standard_deviation, marker='o', capsize=4)
        plt.title(f'{name} Memory Usage vs. Signal Length (Np = 8)')
        plt.xlabel('Signal Length (N)')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.xscale('log', base=2)
        plt.show()


    return average_peak_usage, standard_deviation, signal_length

def get_alpha_resolution(alpha):
    """
    Get the actual cyclic frequency resolution from the alpha vector.
    
    Parameters
    ----------
    alpha : ndarray
        Vector of cyclic frequencies.
    
    Returns
    -------
    delta_alpha : float
        Cyclic frequency resolution.
    """
    # Handle both sorted and fftshifted alpha vectors
    alpha_sorted = np.sort(alpha)
    delta_alpha = np.median(np.diff(alpha_sorted))
    return delta_alpha

import time
import timeit

def run_benchmark_timeit(func, Np, L, fam=False, runs=100):
    # timeit expects a callable with no arguments

    setup_code = """\
import numpy as np 
rng = np.random.default_rng()
test_signal = rng.uniform(-1, 1, 2**15) + rng.uniform(-1, 1, 2**15) * 1j
"""
    
    if fam:
        statement_code = "func(test_signal, Np=Np, L=L)"
    else:
        statement_code = "func(test_signal, Np=Np)"

    timer = timeit.Timer(
        stmt=statement_code,
        setup=setup_code,
        globals={"func": func, "Np": Np, "L": L}
    )
    
    times = timer.repeat(repeat=runs, number=1)

    result = {
        "mean": np.mean(times),
        "stdev": np.std(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }

    print(f"Mean Execution Time: {result['mean']} \n \
    Standard Deviation: {result['stdev']} \n \
    Min: {result['min']} \n \
    Max: {result['max']} ")
    
    return result

def run_all_tests(func_lambda, name="algorithm", Np=64, L=16, N_roc=4096, mode="full", save=False, fam=False):

    print("hello")

    if mode == "full":
        alpha_max = 1.0

        tests = [
            ("Validation Test (Non Conjugate)", lambda: validation_test(func_lambda, name=name, no_of_run=10, alpha_max=alpha_max, conjugate=False, save=save, fam=fam)),
            ("Validation Test (Conjugate)", lambda: validation_test(func_lambda, name=name, no_of_run=10, alpha_max=alpha_max, conjugate=True, save=save, fam=fam)),
            ("Memory Test", lambda: memory_test(func_lambda, name=name, Np=Np, L=L, fam=fam)),
            ("Speed Test (Benchmark)", lambda: run_benchmark_timeit(func_lambda, Np, L, fam=fam)),
            ("Cycle Leakage", lambda: window_test(func_lambda, name=name, Np=Np, L=L, snr=np.arange(-10, 0), fam=fam)),
            ("ROC Plot", lambda: plot_roc_full(func_lambda, Np=Np, L=L, fam=fam, name=name, N=N_roc, no_of_simulation=500, snr=[0, -5, -10]))
        ]
    elif mode == "limited":
        alpha_max = 0.5

        tests = [
            ("Validation Test (Non Conjugate)", lambda: validation_test(func_lambda, name=name, alpha_max=alpha_max, no_of_run=10, conjugate=False, save=save, fam=fam)),
            ("Memory Test", lambda: memory_test(func_lambda, name=name, Np=Np, L=L, fam=fam)),
            ("Speed Test (Benchmark)", lambda: run_benchmark_timeit(func_lambda, Np, L, fam=fam)),
            ("Cycle Leakage", lambda: window_test(func_lambda, name=name, Np=Np, L=L, snr=np.arange(-10, 0), fam=fam)),
            ("ROC Plot", lambda: plot_roc_limited(func_lambda, Np=Np, L=L, fam=fam, name=name, N=N_roc, no_of_simulation=500, snr=[0, -5, -10]))
        ]
    else:
        return ValueError("Mode must be full or limited.")
    
    results = {}

    total_time = 0
    passed = 0
    
    print("=" * 50)
    print("Starting Test Suite")
    print("=" * 50)
    
    for test_name, test_func in tests:
        print(f"\n▶ Running: {test_name}...")
        start = time.perf_counter()
        
        try:
            result = test_func()
            elapsed = time.perf_counter() - start

            total_time += elapsed

            results[test_name] = result
            print(f"  ✓ Completed in {elapsed:.3f}s")

            passed += 1
        except Exception as e:
            elapsed = time.perf_counter() - start
            results[test_name] = result

            total_time += elapsed

            print(f"  ✗ Failed after {elapsed:.3f}s: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{len(tests)} | Total time: {total_time:.3f}s")
    
    return results