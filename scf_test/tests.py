import glob
import json
import os
import re
import subprocess
import time
import timeit
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter
from scipy.signal import find_peaks
from scipy.signal.windows import get_window
from sklearn import metrics

from .psk import create_srrc_qpsk_signal, create_rect_bpsk_signal, create_srrc_bpsk_signal
from .fsk import create_gmsk_signal, create_cpfsk_signal
from .visualization.time_domain import plot_signal

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

def validation_test(func_lambda, name="algorithm", Np=512, snr=10, max_log_2=20, no_of_run=1, alpha_max = 1.0, conjugate=False, plot=True, save=True, fam=False, rng=None, extra_samples=0, complex_rmse=False):
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

    # Only filled when complex_rmse is set: squared error of the complex values, and the points compared
    square_error_complex = np.zeros(range_i)
    num_points_complex = np.zeros(range_i)

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
            # Pass rng for a reproducible run; otherwise every trial is freshly seeded
            trial_rng = np.random.default_rng() if rng is None else rng
            # Noise first then add cfo
            bpsk_signal = create_rect_bpsk_signal(number_of_symbols, samples_per_symbol, rng=trial_rng)
            bpsk_signal_cfo = add_cfo(bpsk_signal, cfo) # Add 0.5Hz CFO
            bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr, rng=trial_rng)
            # extra_samples lets an estimator whose cycle frequency resolution is not 1/N reach it,
            # e.g. Np - 1 for the Fast-SC, whose resolution is 1/(number of samples - Np + 1)
            bpsk_signal_noise_cfo = bpsk_signal_noise_cfo[:N + extra_samples]

            if fam:
                spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, L=Np/4, conjugate=conjugate)
            else:
                spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, conjugate=conjugate)
            if complex_rmse:
                if not np.iscomplexobj(spectral_corr):
                    raise ValueError("The estimator returns magnitudes only, so its complex values cannot be compared.")
                spectral_corr_complex = spectral_corr
            spectral_corr = np.abs(spectral_corr)  # handle complex output

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

                if complex_rmse:
                    # The phase of the SCD at a cycle frequency is set by the time origin, which each
                    # estimator places differently, so that one phase is fitted and removed first.
                    # What is left compares the complex values, including the sign changes across f.
                    scd_slice_complex = spectral_corr_complex[mask & mask_principal]
                    scd_ref_signed = bpsk_scf_signed(f_slice, cycle_frequency_check[k], samples_per_symbol, cfo, conjugate=conjugate)
                    rotation = np.exp(-1j*np.angle(np.vdot(scd_ref_signed, scd_slice_complex)))
                    square_error_complex[i - start_i] += np.sum(np.abs(scd_slice_complex*rotation - scd_ref_signed)**2)
                    num_points_complex[i - start_i] += len(scd_slice_complex)

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

    if complex_rmse:
        return scd_rmse, signal_length, np.sqrt(square_error_complex/num_points_complex)

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
        fig.colorbar(im, ax=ax, label='RMSE')

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
            return im

        # Create and save the animation
        ani = animation.FuncAnimation(fig, update, frames=11, interval=750, blit=True)

        if save:
            ani.save(f'{name}_rmse_animation.mp4', writer='ffmpeg') # Or 'ffmpeg' for mp4

        plt.show()

    return rmse_mat, signal_length_mat

def rmse_vs_theoretical_bpsk(spectral_correlation, f, alpha, samples_per_symbol, cfo, conjugate=False):
    spectral_correlation = np.abs(np.asarray(spectral_correlation))  # handle complex output
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


def plot_roc_non_conjugate(func_lambda, Np=8, no_of_simulation=1000, threshold_resolution=10000, snr=[0], N=2**12, fam=False, L=2, name="roc", save=False):
    # Create a reference CDP using no noise bpsk so we know the cycle frequency
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz
    cfo = 0.05
    srrc_filter_span = 21
    beta = 0.1
    bandwidth_time = 0.3

    bpsk_cf = 4 * 1/samples_per_symbol
    qpsk_cf = 1/samples_per_symbol
    gmsk_cf = 1/samples_per_symbol

    print("Starting non-conjugate ROC test...")
    print(f"  Rect BPSK cycle frequency: α = {bpsk_cf:.4f}")
    print(f"  SRRC QPSK cycle frequency: α = {qpsk_cf:.4f}")
    print(f"  GMSK cycle frequency:      α = {gmsk_cf:.4f}")
        
    average_pf_bpsk = np.zeros((len(snr), threshold_resolution))
    average_pd_bpsk = np.zeros((len(snr), threshold_resolution))

    average_pf_qpsk = np.zeros((len(snr), threshold_resolution))
    average_pd_qpsk = np.zeros((len(snr), threshold_resolution))

    average_pf_gmsk = np.zeros((len(snr), threshold_resolution))
    average_pd_gmsk = np.zeros((len(snr), threshold_resolution))

    roc_auc = np.zeros(len(snr))
    
    
    for i in range(len(snr)):
        signal_flag = True
        # Statistic of every trial, per signal: thresholds are set afterwards over its actual range,
        # since the coherence can exceed 1 (e.g. with a PSD that does not match the numerator)
        present, absent = ([], [], []), ([], [], [])

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
            spectral_coherence_bpsk_nc = np.abs(spectral_coherence_bpsk_nc)  # handle complex output
            spectral_coherence_qpsk_nc = np.abs(spectral_coherence_qpsk_nc)
            spectral_coherence_gmsk_nc = np.abs(spectral_coherence_gmsk_nc)

            coh_max_bpsk = []
            coh_max_qpsk = []
            coh_max_gmsk = []

            # alpha binning. Assuming cycle frequency resolution is 1/N
            # alpha = np.round(alpha * N) / N

            # New: does not assume alpha resolution is 1/N
            if j == 0:
                resolution = get_alpha_resolution(alpha) / 2 
            
            # Extract maximum coh at alpha = 0.4 for non-conj BPSK SCD
            mask_pos_4_nc_bpsk = np.abs(alpha - bpsk_cf) <= resolution
            coh_max_bpsk.append(np.max(spectral_coherence_bpsk_nc[mask_pos_4_nc_bpsk]))

            # Extract maximum coh at alpha = 0.1 for non-conj QPSK SCD
            mask_pos_1_nc_qpsk = np.abs(alpha - qpsk_cf) <= resolution 
            coh_max_qpsk.append(np.max(spectral_coherence_qpsk_nc[mask_pos_1_nc_qpsk]))

            # Extract maximum coh at alpha = 0.1 for non-conj GMSK SCD
            mask_pos_1_nc_gmsk = np.abs(alpha - gmsk_cf) <= resolution 
            coh_max_gmsk.append(np.max(spectral_coherence_gmsk_nc[mask_pos_1_nc_gmsk]))

            coh_max_bpsk = np.array(coh_max_bpsk).reshape((-1, 1))
            coh_max_qpsk = np.array(coh_max_qpsk).reshape((-1, 1))
            coh_max_gmsk = np.array(coh_max_gmsk).reshape((-1, 1))

            for store, value in zip(present if signal_flag else absent, (coh_max_bpsk, coh_max_qpsk, coh_max_gmsk)):
                store.append(float(value[0, 0]))

        # len(coh_max) is only relevant if you want to check other alphas
        top = max(1.0, max(max(v) for v in present + absent)) * (1 + 1e-9)
        threshold = np.linspace(0, top, threshold_resolution).reshape((1, -1))
        average_pd_bpsk[i, :] = np.mean(np.array(present[0])[:, None] >= threshold, axis=0)
        average_pf_bpsk[i, :] = np.mean(np.array(absent[0])[:, None] >= threshold, axis=0)
        average_pd_qpsk[i, :] = np.mean(np.array(present[1])[:, None] >= threshold, axis=0)
        average_pf_qpsk[i, :] = np.mean(np.array(absent[1])[:, None] >= threshold, axis=0)
        average_pd_gmsk[i, :] = np.mean(np.array(present[2])[:, None] >= threshold, axis=0)
        average_pf_gmsk[i, :] = np.mean(np.array(absent[2])[:, None] >= threshold, axis=0)
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
        
        print(f"{snr[i]}dB SNR Score: {roc_auc_total[i]:.3f}/3")

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space
    if save:
        plt.savefig("fig/" + name + "_roc_limited" + ".png")
        np.save("data/roc_test_" + name + "_score_non_conjugate", roc_auc_total)
    plt.show()

    average_pd = np.concatenate((average_pd_bpsk, average_pd_qpsk, average_pd_gmsk) ,axis=1)
    average_pf = np.concatenate((average_pf_bpsk, average_pf_qpsk, average_pf_gmsk) ,axis=1)
    
    return average_pd, average_pf, roc_auc, threshold_resolution

def plot_roc_conjugate(func_lambda, Np=8, no_of_simulation=1000, threshold_resolution=10000, snr=[0], N=2**12, fam=False, L=2, name="roc", save=False):
    # Create a reference CDP using no noise bpsk so we know the cycle frequency
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz
    cfo = 0.05
    srrc_filter_span = 21
    beta = 0.1

    bpsk_cf = 2*cfo + 4 * 1/samples_per_symbol
    msk_cf = cfo + 2 * 1/samples_per_symbol
    srrc_bpsk_cf = 2*cfo + 1/samples_per_symbol

    print("Starting conjugate ROC test...")
    print(f"  Rect BPSK cycle frequency: α = {bpsk_cf:.4f}")
    print(f"  MSK cycle frequency:       α = {msk_cf:.4f}")
    print(f"  SRRC BPSK cycle frequency: α = {srrc_bpsk_cf:.4f}")
        
    average_pf_bpsk = np.zeros((len(snr), threshold_resolution))
    average_pd_bpsk = np.zeros((len(snr), threshold_resolution))

    average_pf_msk = np.zeros((len(snr), threshold_resolution))
    average_pd_msk = np.zeros((len(snr), threshold_resolution))

    average_pf_srrc_bpsk = np.zeros((len(snr), threshold_resolution))
    average_pd_srrc_bpsk = np.zeros((len(snr), threshold_resolution))

    roc_auc = np.zeros(len(snr))
    
    
    for i in range(len(snr)):
        signal_flag = True
        # Statistic of every trial, per signal: thresholds are set afterwards over its actual range,
        # since the coherence can exceed 1 (e.g. with a PSD that does not match the numerator)
        present, absent = ([], [], []), ([], [], [])

        for j in range(no_of_simulation*2):
            # Half signal + noise, half only noise
            bpsk_signal = create_rect_bpsk_signal(N, samples_per_symbol)
            bpsk_signal_cfo = add_cfo(bpsk_signal, cfo) # Add 0.5Hz CFO
            bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr[i])

            msk_signal, _, _ = create_cpfsk_signal(N, samples_per_symbol, modulation_index=0.5, f_carrier=cfo) 
            msk_signal_noise_cfo = add_awgn_snr(msk_signal, desired_snr=snr[i])

            srrc_bpsk_signal = add_cfo(create_srrc_bpsk_signal(N, samples_per_symbol, srrc_filter_span, beta), cfo)
            srrc_bpsk_signal_noise_cfo = add_awgn_snr(srrc_bpsk_signal, desired_snr=snr[i])
            
            # Even for signal present
            if j % 2 == 0:
                test_signal_bpsk = bpsk_signal_noise_cfo[:N]
                test_signal_msk = msk_signal_noise_cfo[:N]
                test_signal_srrc_bpsk = srrc_bpsk_signal_noise_cfo[:N]
                signal_flag = True
            # Odd for noise only
            else:
                test_signal_bpsk = (bpsk_signal_noise_cfo - bpsk_signal_cfo)[:N]
                test_signal_msk = (msk_signal_noise_cfo - msk_signal)[:N]
                test_signal_srrc_bpsk = (srrc_bpsk_signal_noise_cfo - srrc_bpsk_signal)[:N]
                signal_flag = False

            if fam:
                spectral_coherence_bpsk_c, _, alpha = func_lambda(test_signal_bpsk, Np=Np, L=L, conjugate=True, coherence=True)

                spectral_coherence_msk_c, _, _ = func_lambda(test_signal_msk, Np=Np, L=L, conjugate=True, coherence=True)

                spectral_coherence_srrc_bpsk_c, _, _ = func_lambda(test_signal_srrc_bpsk, Np=Np, L=L, conjugate=True, coherence=True)
            else:
                spectral_coherence_bpsk_c, _, alpha = func_lambda(test_signal_bpsk, Np=Np, conjugate=True, coherence=True)

                spectral_coherence_msk_c, _, _ = func_lambda(test_signal_msk, Np=Np, conjugate=True, coherence=True)

                spectral_coherence_srrc_bpsk_c, _, _ = func_lambda(test_signal_srrc_bpsk, Np=Np, conjugate=True, coherence=True)
            spectral_coherence_bpsk_c = np.abs(spectral_coherence_bpsk_c)  # handle complex output
            spectral_coherence_msk_c  = np.abs(spectral_coherence_msk_c)
            spectral_coherence_srrc_bpsk_c = np.abs(spectral_coherence_srrc_bpsk_c)

            coh_max_bpsk = []
            coh_max_msk = []
            coh_max_srrc_bpsk = []

            # alpha binning. Assuming cycle frequency resolution is N
            # alpha = np.round(alpha * N) / N
            # after, that it would be np.abs(alpha - desired_alpha) <= 1/(2*N) to get the slice

            # New: does not assume alpha resolution is 1/N
            if j == 0:
                resolution = get_alpha_resolution(alpha) / 2 
            
            # Extract maximum coh at alpha = 0.5 for conj BPSK SCD
            mask_pos_5_c_bpsk = np.abs(alpha - bpsk_cf) <= resolution
            coh_max_bpsk.append(np.max(spectral_coherence_bpsk_c[mask_pos_5_c_bpsk]))

            # Extract maximum coh at alpha = 0.4 for non-conj MSK SCD
            mask_pos_1_nc_msk = np.abs(alpha - msk_cf) <= resolution
            coh_max_msk.append(np.max(spectral_coherence_msk_c[mask_pos_1_nc_msk]))

            # Extract maximum coh at alpha = 0.2 for conj SRRC BPSK SCD
            mask_pos_2_c_srrc_bpsk = np.abs(alpha - srrc_bpsk_cf) <= resolution
            coh_max_srrc_bpsk.append(np.max(spectral_coherence_srrc_bpsk_c[mask_pos_2_c_srrc_bpsk]))

            coh_max_bpsk = np.array(coh_max_bpsk).reshape((-1, 1))
            coh_max_msk = np.array(coh_max_msk).reshape((-1, 1))
            coh_max_srrc_bpsk = np.array(coh_max_srrc_bpsk).reshape((-1, 1))

            for store, value in zip(present if signal_flag else absent, (coh_max_bpsk, coh_max_msk, coh_max_srrc_bpsk)):
                store.append(float(value[0, 0]))

        # len(coh_max) is only relevant if you want to check other alphas
        top = max(1.0, max(max(v) for v in present + absent)) * (1 + 1e-9)
        threshold = np.linspace(0, top, threshold_resolution).reshape((1, -1))
        average_pd_bpsk[i, :] = np.mean(np.array(present[0])[:, None] >= threshold, axis=0)
        average_pf_bpsk[i, :] = np.mean(np.array(absent[0])[:, None] >= threshold, axis=0)
        average_pd_msk[i, :] = np.mean(np.array(present[1])[:, None] >= threshold, axis=0)
        average_pf_msk[i, :] = np.mean(np.array(absent[1])[:, None] >= threshold, axis=0)
        average_pd_srrc_bpsk[i, :] = np.mean(np.array(present[2])[:, None] >= threshold, axis=0)
        average_pf_srrc_bpsk[i, :] = np.mean(np.array(absent[2])[:, None] >= threshold, axis=0)
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

        roc_auc_msk = np.round(metrics.auc(average_pf_msk[i, :], average_pd_msk[i, :]), 3)
        axs[1].plot(average_pf_msk[i, :], average_pd_msk[i, :], label=f"{snr[i]}dB SNR")
        axs[1].plot([0, 1], [0, 1], color='red', linestyle='--')
        # axs[1, j].legend()

        roc_auc_srrc_bpsk = np.round(metrics.auc(average_pf_srrc_bpsk[i, :], average_pd_srrc_bpsk[i, :]), 3)
        axs[2].plot(average_pf_srrc_bpsk[i, :], average_pd_srrc_bpsk[i, :], label=f"{snr[i]}dB SNR")
        axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')
        # axs[2, j].legend()

        axs[0].set_title("Conjugate Rect BPSK")
        axs[1].set_title("Conjugate MSK")
        axs[2].set_title("Conjugate SRRC BPSK")

        roc_auc_total[i] += roc_auc_bpsk + roc_auc_msk + roc_auc_srrc_bpsk

        roc_auc[i, 0] = roc_auc_bpsk
        roc_auc[i, 1] = roc_auc_msk
        roc_auc[i, 2] = roc_auc_srrc_bpsk
        
        print(f"{snr[i]}dB SNR Score: {roc_auc_total[i]:.3f}/3")

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space
    if save:
        plt.savefig("fig/" + name + "_roc_limited" + ".png")
        np.save("data/roc_test_" + name + "_score_conjugate", roc_auc_total)

    plt.show()

    average_pd = np.concatenate((average_pd_bpsk, average_pd_msk, average_pd_srrc_bpsk) ,axis=1)
    average_pf = np.concatenate((average_pf_bpsk, average_pf_msk, average_pf_srrc_bpsk) ,axis=1)
    
    return average_pd, average_pf, roc_auc, threshold_resolution

def extended_window_test(scf_func, name="algorithm_window_test", signal_length=4096, Np=64, L=16, fam=False, conjugate=False, snr=0, number_of_runs=100, plot=False):
    number_of_symbols = int(signal_length/8)
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz

    N = signal_length

    # Following Eric April's recommendation
    # a(n) or first window -> Large attenuation
    # g(n) or second window -> Small bandwidth
    #  "hamming", "hann", "cosine", "barthann", "lanczos" are middle of the road windows
    # small_bandwidth_window = ["boxcar", "triang", ("kaiser", 4), "hamming", "hann", "cosine", "barthann", "lanczos", ("tukey", 0.2)]
    # large_attenuation_window = [("chebwin", 100), ("kaiser", 14), "blackmanharris", "nuttall", "flattop", "parzen", "bohman", "taylor", "hamming", "hann", "cosine", "barthann", "lanczos"]

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
                if fam:
                    scd, _, alpha = scf_func(test_signal, Np=Np, L=L, window_1=all_windows[i], window_2=all_windows[j], conjugate=conjugate)
                else:
                    scd, _, alpha = scf_func(test_signal, Np=Np, window_1=all_windows[i], window_2=all_windows[j], conjugate=conjugate)
                scd = np.abs(scd)  # handle complex output

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

def window_test(scf_func, name="algorithm", signal_length=4096, Np=64, L=16,
                conjugate=False, snr=[0], number_of_runs=100, fam=False,
                plot=True, outlier_threshold=100, verbose=True):
    """Cycle leakage test with automatic robustness to spurious outliers.

    For each SNR, computes per-run cycle leakage, then reports the mean
    unless spurious high-magnitude runs are detected (e.g. S3CA's
    sparse-FFT blowups), in which case the median is used instead.
    """
    number_of_symbols = int(signal_length / 8)
    samples_per_symbol = 10  # Symbol rate = 0.1 Hz
    N = signal_length
    average_cycle_leakage = np.zeros(len(snr))

    for i in range(len(snr)):
        per_run = []

        for j in range(number_of_runs):
            bpsk_signal = create_rect_bpsk_signal(number_of_symbols, samples_per_symbol)
            bpsk_signal_cfo = add_cfo(bpsk_signal, 0.05)
            bpsk_signal_cfo_noise = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr[i])
            test_signal = bpsk_signal_cfo_noise[:signal_length]

            if fam:
                scd, _, alpha = scf_func(test_signal, Np=Np, L=L,
                                         conjugate=conjugate, coherence=False)
            else:
                scd, _, alpha = scf_func(test_signal, Np=Np,
                                         conjugate=conjugate, coherence=False)
            scd = np.abs(scd)  # in case output is complex

            if not conjugate:
                mask = (np.abs(alpha) > 1 / N) & (np.abs(alpha) < 0.1 - 1 / N)
            else:
                mask = (np.abs(alpha - 0.1) > 1 / N) & (np.abs(alpha - 0.1) < 0.1 - 1 / N)

            if mask.sum() > 0:
                val = np.sum(scd[mask]) / mask.sum()
            else:
                val = np.nan
            per_run.append(val)

        per_run = np.array(per_run)

        # print(f"percentiles:")
        # for p in [10, 25, 50, 75, 90, 95]:
        #     print(f"  {p}th: {np.percentile(per_run[~np.isnan(per_run)], p):.3f}")
        # print(f"values under 1: {(per_run < 1).sum()}")
        # print(f"values 1-10:    {((per_run >= 1) & (per_run < 10)).sum()}")
        # print(f"values 10-100:  {((per_run >= 10) & (per_run < 100)).sum()}")
        # print(f"values >100:    {(per_run > 100).sum()}")

        # Decide mean vs median based on presence of outliers
        n_outliers = np.sum(per_run > outlier_threshold)
        n_nan = np.sum(np.isnan(per_run))

        if n_outliers > 0 or n_nan > 0:
            # Spurious blowups present (e.g. S3CA): use median for robustness
            average_cycle_leakage[i] = np.nanmedian(per_run)
            stat_used = "median"
        else:
            average_cycle_leakage[i] = np.nanmean(per_run)
            stat_used = "mean"

        if verbose:
            print(f"SNR={snr[i]} dB | {stat_used}: {average_cycle_leakage[i]:.3f} | "
                  f"median: {np.nanmedian(per_run):.3f}, "
                  f"mean: {np.nanmean(per_run):.3e}, "
                  f"outliers(>{outlier_threshold}): {n_outliers}, "
                  f"nan: {n_nan}")

    if plot:
        fig = plt.figure(figsize=(5, 3))
        plt.plot(snr, average_cycle_leakage, 'x--')
        plt.title(f"{name} Cycle Leakage: {np.mean(average_cycle_leakage):.3f}")
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


def _format_seconds(value, _pos=None):
    """Tick label for a time in seconds, with just enough decimals.

    0.001 -> "0.001", 0.01 -> "0.01", 1 -> "1", 10 -> "10". Keeps the whole
    axis in one unit rather than switching between us/ms/s.
    """
    if value <= 0:
        return ''
    decimals = max(0, int(np.ceil(-np.log10(value))))
    return f'{value:.{decimals}f}'


def plot_benchmark(df, param_name='param', title=None, xlabel=None, stat='min',
                   log_x=True, log_y=True, figsize=(6, 4), save_path=None):
    """Plot benchmark results.

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
    stat : {'min', 'median', 'mean', 'max'}
        Timing statistic to plot. 'min' (the default) is the least
        noise-contaminated estimate and the usual choice for benchmarks.
        The y-axis label and the error bars follow this choice — see below.
    log_x, log_y : bool
        Whether to use log scale. When True and data spans less than
        one decade, minor ticks are labeled so the axis isn't blank.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Notes
    -----
    Error bars are only drawn where they mean something for the chosen
    statistic: stddev around 'mean', the asymmetric q1..q3 interquartile
    range around 'median', and none at all for 'min'/'max' (an extremum
    has no symmetric spread, and min - stddev is frequently negative,
    which cannot be drawn on a log axis).
    """
    if df is None or df.empty:
        print("No data to plot!")
        return

    if stat not in df.columns:
        print(f"Column '{stat}' not in DataFrame — available: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for algo_name in df['algorithm'].unique():
        subset = df[df['algorithm'] == algo_name]

        if stat == 'mean':
            yerr = subset['stddev']
        elif stat == 'median':
            yerr = [subset['median'] - subset['q1'], subset['q3'] - subset['median']]
        else:
            yerr = None

        ax.errorbar(subset[param_name], subset[stat],
                    marker='o', linestyle='-', label=algo_name,
                    yerr=yerr, capsize=4)

    ax.set_xlabel(xlabel or param_name)
    ax.set_ylabel(f'{stat.capitalize()} Execution Time (seconds)')
    ax.set_title(title or 'Benchmark Comparison')
 
    # --- X axis ----------------------------------------------------------
    if log_x:
        ax.set_xscale('log', base=2)
        # Plain integers instead of 2^n notation for the x axis (param
        # values are typically already powers of 2 and read more clearly
        # as 8, 16, 32, ... than as 2^3, 2^4, 2^5, ...).
        ax.xaxis.set_major_formatter(ScalarFormatter())
 
    # --- Y axis ----------------------------------------------------------
    if log_y:
        # Base 10, not base 2: times are raw seconds straight from the
        # pytest-benchmark CSV, so decade ticks (0.001, 0.01, 0.1, 1, 10)
        # keep one unit across the whole axis and make each step a plain
        # x10. Base 2 put the ticks on powers of two, which needed SI
        # prefixes to stay short and mixed us/ms/s on a single axis.
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_seconds))
        # If all data sits inside a single decade (e.g. all values between
        # 2 s and 4 s), matplotlib's default LogLocator puts no major ticks
        # in the visible range and the axis comes out blank. Detect this
        # from the data span and add labeled minor ticks only when needed —
        # otherwise the minors clutter wide-range plots like
        # (fam + ssca + scf_2d_fft) together.
        y_min = float(df[stat].min())
        y_max = float(df[stat].max())
        span_decades = np.log10(y_max / y_min) if y_min > 0 else 10.0
        if span_decades < 1.0:
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='all'))
            ax.yaxis.set_minor_formatter(FuncFormatter(_format_seconds))
            ax.tick_params(axis='y', which='minor', labelsize=8)
 
    # Major ticks only. On a log axis which="both" adds the ~8 minor lines
    # packed into every decade, which reads as clutter rather than a grid.
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.4)
    ax.legend()
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")
 
    plt.show()
def memory_test(func_lambda, name="algorithm", Np=8, L=2, max_log_2=18, no_of_run=1, fam=False, plot=True):
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
        plt.title(f'{name} Memory Usage vs. Signal Length (Np = {Np})')
        plt.xlabel('Signal Length (N)')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.xscale('log', base=2)
        plt.show()


    return average_peak_usage, standard_deviation, signal_length

def run_benchmark_timeit(func, Np, L, fam=False, runs=10):
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

def run_all_tests(func_lambda, name="algorithm", Np=64, L=16, N_roc=4096,
                  mode="full", save=False, fam=False, skip=None):
    """Run the standard test battery against an SCF estimator.

    Parameters
    ----------
    func_lambda, name, Np, L, N_roc, mode, save, fam
        Unchanged from the previous signature.
    skip : str, iterable of str, or None
        Names (or substrings) of tests to skip. Matching is case-
        insensitive substring. Examples:

            skip="validation"                # skip both validation tests
            skip=["validation", "memory"]    # skip all three
            skip="roc (conjugate)"           # skip one specific test
            skip=None                        # run everything (default)

        Useful for estimators whose memory or runtime footprint makes
        certain tests impractical — e.g. the 2D-FFT SCF's O(N^2) SACM
        at N_roc=4096 OOMs on 8 GB machines during the ROC sweep.
    """
    if mode == "full":
        alpha_max = 1.0
        tests = [
            ("Validation Test (Non Conjugate)", lambda: validation_test(func_lambda, name=name, no_of_run=10, alpha_max=alpha_max, conjugate=False, save=save, fam=fam)),
            ("Validation Test (Conjugate)", lambda: validation_test(func_lambda, name=name, no_of_run=10, alpha_max=alpha_max, conjugate=True, save=save, fam=fam)),
            ("Phase Structure Test (Non Conjugate)", lambda: phase_structure_test(func_lambda, name=name, alpha_max=alpha_max, conjugate=False, fam=fam)),
            ("Phase Structure Test (Conjugate)", lambda: phase_structure_test(func_lambda, name=name, alpha_max=alpha_max, conjugate=True, fam=fam)),
            ("Coherence Validation (Non Conjugate)", lambda: coherence_validation_test(func_lambda, name=name, no_of_run=10, alpha_max=alpha_max, conjugate=False, save=save, fam=fam)),
            ("Coherence Validation (Conjugate)", lambda: coherence_validation_test(func_lambda, name=name, no_of_run=10, alpha_max=alpha_max, conjugate=True, save=save, fam=fam)),
            ("CDP Test (Non Conjugate)", lambda: cyclic_domain_profile_test(func_lambda, Np=Np, L=L, conjugate=False, fam=fam)),
            ("CDP Test (Conjugate)", lambda: cyclic_domain_profile_test(func_lambda, Np=Np, L=L, conjugate=True, fam=fam)),
            ("Memory Test", lambda: memory_test(func_lambda, name=name, Np=Np, L=L, fam=fam)),
            ("Speed Test (Benchmark)", lambda: run_benchmark_timeit(func_lambda, Np, L, fam=fam)),
            ("Cycle Leakage", lambda: window_test(func_lambda, name=name, Np=Np, L=L, snr=np.arange(-10, 0), fam=fam)),
            ("ROC Plot (Non Conjugate)", lambda: plot_roc_non_conjugate(func_lambda, Np=Np, L=L, fam=fam, name=name, N=N_roc, no_of_simulation=500, snr=[0, -5, -10])),
            ("ROC Plot (Conjugate)", lambda: plot_roc_conjugate(func_lambda, Np=Np, L=L, fam=fam, name=name, N=N_roc, no_of_simulation=500, snr=[0, -5, -10])),
        ]
    elif mode == "limited":
        alpha_max = 0.5
        tests = [
            ("Validation Test (Non Conjugate)", lambda: validation_test(func_lambda, name=name, alpha_max=alpha_max, no_of_run=10, conjugate=False, save=save, fam=fam)),
            ("Phase Structure Test (Non Conjugate)", lambda: phase_structure_test(func_lambda, name=name, alpha_max=alpha_max, conjugate=False, fam=fam)),
            ("Coherence Validation (Non Conjugate)", lambda: coherence_validation_test(func_lambda, name=name, Np=Np, no_of_run=10, alpha_max=alpha_max, conjugate=False, save=save, fam=fam)),
            ("CDP Test (Non Conjugate)", lambda: cyclic_domain_profile_test(func_lambda, Np=Np, L=L, conjugate=False, fam=fam)),
            ("Memory Test", lambda: memory_test(func_lambda, name=name, Np=Np, L=L, fam=fam)),
            ("Speed Test (Benchmark)", lambda: run_benchmark_timeit(func_lambda, Np, L, fam=fam)),
            ("Cycle Leakage", lambda: window_test(func_lambda, name=name, Np=Np, L=L, snr=np.arange(-10, 0), fam=fam)),
            ("ROC Plot (Non Conjugate)", lambda: plot_roc_non_conjugate(func_lambda, Np=Np, L=L, fam=fam, name=name, N=N_roc, no_of_simulation=500, snr=[0, -5, -10])),
        ]
    else:
        raise ValueError("Mode must be full or limited.")

    # --- normalise skip -> list of lowercased patterns --------------------
    if skip is None:
        skip_patterns = []
    elif isinstance(skip, str):
        skip_patterns = [skip.lower()]
    else:
        skip_patterns = [str(s).lower() for s in skip]

    def _should_skip(test_name: str) -> bool:
        low = test_name.lower()
        return any(pat in low for pat in skip_patterns)

    # --- partition into run list + skip list ------------------------------
    to_run = [(n, f) for n, f in tests if not _should_skip(n)]
    skipped = [n for n, _ in tests if _should_skip(n)]

    results = {}
    total_time = 0.0
    passed = 0
    failed = 0

    print("=" * 50)
    print("Starting Test Suite")
    print("=" * 50)
    if skipped:
        print(f"Skipping {len(skipped)} test(s) by request:")
        for n in skipped:
            print(f"  - {n}")
            results[n] = None   # explicit marker in the results dict
        print()

    for test_name, test_func in to_run:
        print(f"\n▶ Running: {test_name}...")
        start = time.perf_counter()
        result = None  # guarantee defined for the except branch

        try:
            result = test_func()
            elapsed = time.perf_counter() - start
            total_time += elapsed
            results[test_name] = result
            print(f"  ✓ Completed in {elapsed:.3f}s")
            passed += 1
        except Exception as e:
            elapsed = time.perf_counter() - start
            total_time += elapsed
            results[test_name] = result
            failed += 1
            print(f"  ✗ Failed after {elapsed:.3f}s: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Passed:  {passed}/{len(to_run)}")
    if failed:
        print(f"Failed:  {failed}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
    print(f"Total time: {total_time:.3f}s")

    return results

def spectral_correlation_to_coherence(psd, spectral_correlation, f, alpha, conjugate=False, fs=1):
    N = len(psd)

    psd = np.fft.ifftshift(psd)

    sample_1 = np.rint((f + alpha/2) * N/fs).astype(int)
    if conjugate:
        sample_2 = np.rint((alpha/2 - f) * N/fs).astype(int)
    else:
        sample_2 = np.rint((f - alpha/2) * N/fs).astype(int)

    coherence_denominator = np.sqrt(psd[sample_1] * psd[sample_2])

    coherence = spectral_correlation/coherence_denominator

    return coherence

def calculate_bpsk_scf(number_of_points, alpha_slices, Tk, f_offset, conjugate=False, coherence=False, plot=False):
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
        significant_slices = 1/Tk * np.arange(-20, 20)

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

    if coherence:
        psd = calculate_bpsk_scf(number_of_points, [0], Tk, f_offset, conjugate=False).flatten() + 0.1
        spectral_coherence = spectral_correlation_to_coherence(psd, spectral_correlation, f_grid, alpha_grid, conjugate=conjugate, fs=1)
        return spectral_coherence
    else:
        return spectral_correlation


def bpsk_scf_signed(f, alpha, Tk, f_offset, conjugate=False):
    """
    Theoretical SCD of rectangular-pulse BPSK at the points (f, alpha), keeping its sign.

    The result is real. The complete SCD is this multiplied by one phase per cycle
    frequency, set by the symbol timing (and by the carrier phase for the conjugate
    SCD). calculate_bpsk_scf gives the magnitude of the same expression on a grid.
    """
    term1 = Tk * np.sinc((f - f_offset + alpha/2) * Tk)
    if conjugate:
        term2 = Tk * np.sinc((alpha/2 - f - f_offset) * Tk)
    else:
        term2 = Tk * np.sinc((f - f_offset - alpha/2) * Tk)

    return 1/Tk * term1 * term2

def phase_structure_test(func_lambda, name="algorithm", Np=128, signal_length=4096, snr=30, no_of_run=5, alpha_max=1.0, conjugate=False, fam=False, extra_samples=0, threshold=0.2, rng=None):
    """
    Check the sign structure of the SCD estimate, independently of its scale.

    A scale-free diagnostic: validation_test(complex_rmse=True) is the stricter check, since it
    fits only the phase and so also tests the magnitude.

    For rectangular BPSK the SCD at each cycle frequency is a real, signed function of f
    times one complex constant. The estimate at every cycle frequency is fitted with that
    constant; the residual of the fit, relative to the size of the estimate, is small only
    if the sign of the estimate changes across f where the theory says it does.

    Returns
    -------
    passed : bool
        Whether the residual pooled over the cycle frequencies is below threshold
    result : dict
        The pooled residual, the same for a magnitude-only fit, and the residual and
        fitted phase (degrees) at each cycle frequency
    """
    samples_per_symbol = 10 # Symbol rate = 0.1 Hz
    cfo = 0.05
    N = signal_length

    if conjugate:
        cycle_frequency_check = 1/(samples_per_symbol) * np.arange(-30, 30) + 2*cfo
    else:
        cycle_frequency_check = 1/(samples_per_symbol) * np.arange(1, 30)

    cycle_frequency_check = np.round(cycle_frequency_check[np.abs(cycle_frequency_check) < alpha_max], 2)

    error = np.zeros(len(cycle_frequency_check))
    error_magnitude = np.zeros(len(cycle_frequency_check))
    energy = np.zeros(len(cycle_frequency_check))
    phase = np.zeros((no_of_run, len(cycle_frequency_check)))

    for j in range(no_of_run):
        trial_rng = np.random.default_rng() if rng is None else rng
        bpsk_signal = create_rect_bpsk_signal(N, samples_per_symbol, rng=trial_rng)
        bpsk_signal_cfo = add_cfo(bpsk_signal, cfo)
        bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr, rng=trial_rng)[:N + extra_samples]

        if fam:
            spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, L=Np/4, conjugate=conjugate)
        else:
            spectral_corr, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, conjugate=conjugate)

        if not np.iscomplexobj(spectral_corr):
            raise ValueError("The estimator returns magnitudes only, so its phase cannot be checked.")

        # Same slices as validation_test
        mask_principal = np.abs(f) + np.abs(alpha / 2.0) < 0.5
        alpha = np.round(alpha * N) / N

        for k in range(len(cycle_frequency_check)):
            mask = (np.abs(alpha - cycle_frequency_check[k]) <= 1/(2*N)) & mask_principal
            scd_slice = spectral_corr[mask]
            scd_ref = bpsk_scf_signed(f[mask], cycle_frequency_check[k], samples_per_symbol, cfo, conjugate=conjugate)

            # Least-squares fit of one complex constant, and of one real gain on the magnitudes
            constant = np.vdot(scd_ref, scd_slice) / np.vdot(scd_ref, scd_ref)
            gain = np.vdot(np.abs(scd_ref), np.abs(scd_slice)) / np.vdot(scd_ref, scd_ref)

            error[k] += np.sum(np.abs(scd_slice - constant * scd_ref)**2)
            error_magnitude[k] += np.sum((np.abs(scd_slice) - gain * np.abs(scd_ref))**2)
            energy[k] += np.sum(np.abs(scd_slice)**2)
            phase[j, k] = np.degrees(np.angle(constant))

    residual = np.sqrt(np.sum(error) / np.sum(energy))
    residual_magnitude = np.sqrt(np.sum(error_magnitude) / np.sum(energy))

    print(f"{name} phase structure: residual {residual:.3f} (magnitude only: {residual_magnitude:.3f}), threshold {threshold}")

    return residual < threshold, {
        "residual": residual,
        "residual_magnitude_only": residual_magnitude,
        "cycle_frequencies": cycle_frequency_check,
        "residual_per_cycle_frequency": np.sqrt(error / energy),
        "phase_degrees": phase,
    }

def coherence_validation_test(func_lambda, name="algorithm", Np=512, snr=10, max_log_2=19, no_of_run=1, alpha_max = 1.0, conjugate=False, plot=True, save=True, fam=False):
    start_i = 10 # or 1024 (change if you want higher Np)
    end_i = max_log_2 
    range_i = end_i - start_i
    
    samples_per_symbol = 10 # Symbol rate = 0.10 Hz
    cfo = 0.05

    max_no_of_symbols = 2**(end_i-1); #65536 samples
    max_length = max_no_of_symbols * samples_per_symbol; #262144 samples

    if conjugate:
        cycle_frequency_check = 1/(samples_per_symbol) * np.arange(-30, 30) + 2*cfo;
    else:
        cycle_frequency_check = 1/(samples_per_symbol) * np.arange(1, 30);
    
    cycle_frequency_check = np.round(cycle_frequency_check[np.abs(cycle_frequency_check) < alpha_max], 2)

    coh_ref = calculate_bpsk_scf(max_length, cycle_frequency_check, samples_per_symbol, cfo, conjugate=conjugate, coherence=True)

    # print(np.shape(scd_ref))

    coh_ref = np.fft.ifftshift(coh_ref, axes=1)

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
                spectral_coh, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, L=Np/4, conjugate=conjugate, coherence=True)
            else:
                spectral_coh, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, conjugate=conjugate, coherence=True)
            spectral_coh = np.abs(spectral_coh)  # handle complex output

            # Remove any value outside of the principal diamond
            mask_principal = np.abs(f) + np.abs(alpha / 2.0) < 0.5

            # alpha binning. Assuming cycle frequency resolution is N
            alpha = np.round(alpha * N) / N

            for k in range(len(cycle_frequency_check)):
                mask = np.abs(alpha - cycle_frequency_check[k]) <= 1/(2*N)
                coh_slice = spectral_coh[mask & mask_principal]
                f_slice = f[mask & mask_principal]
                f_index = np.rint(f_slice * max_length).astype(int)
                num_points += len(coh_slice)
                scd_sum_square_error += np.sum((coh_slice - coh_ref[k, f_index])**2)

                if plot and j == no_of_run - 1:
                    axs[k].plot(f_slice, coh_slice, label=name)
                    axs[k].plot(f_slice, coh_ref[k, f_index], label="Theory", color="orange")
                    axs[k].label_outer()
                    axs[k].text(0.9, 0.5, f'alpha = {cycle_frequency_check[k]}', horizontalalignment='center', verticalalignment='center', transform=axs[k].transAxes)
                elif plot:
                    axs[k].plot(f_slice, coh_slice)
                
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

def compute_cdp(Sx, alpha, reduction='max'):
    """Compute the cyclic domain profile I(α) = max_f |S(f, α)|.

    Parameters
    ----------
    Sx : ndarray
        SCF magnitudes (or complex SCF — magnitude is taken internally).
        Any shape; will be flattened.
    alpha : ndarray
        Cyclic-frequency coordinate of each entry in Sx. Must have the
        same shape as Sx.
    reduction : {'max', 'sum'}
        How to reduce over f at each unique α. 'max' is the standard
        definition; 'sum' is less noise-sensitive but loses peak contrast.

    Returns
    -------
    alpha_unique : (K,) ndarray
        Sorted unique α values present in the input.
    profile : (K,) ndarray
        max_f (or Σ_f) of |Sx| at each α.
    """
    Sx_mag = np.abs(np.asarray(Sx)).ravel()
    alpha_flat = np.asarray(alpha).ravel()

    if Sx_mag.shape != alpha_flat.shape:
        raise ValueError(
            f"Sx and alpha must have the same shape after flattening; "
            f"got {Sx_mag.shape} and {alpha_flat.shape}"
        )

    # Quantize to 10 decimal places to avoid floating-point near-duplicates
    # creating spurious distinct α values across different estimators.
    alpha_q = np.round(alpha_flat, decimals=10)
    alpha_unique, inverse = np.unique(alpha_q, return_inverse=True)
    profile = np.zeros_like(alpha_unique, dtype=Sx_mag.dtype)

    if reduction == 'max':
        np.maximum.at(profile, inverse, Sx_mag)
    elif reduction == 'sum':
        np.add.at(profile, inverse, Sx_mag)
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    return alpha_unique, profile


def plot_cdp(estimates, reduction='max', ax=None, log_scale=False,
             alpha_range=None, title=None, normalize=False):
    """Plot cyclic domain profiles for multiple estimators on one axis.

    Parameters
    ----------
    estimates : dict[str, tuple] or list[tuple]
        Maps label -> (Sx, alpha), or a list of (label, Sx, alpha) triples.
        Sx and alpha are the algorithm's native output arrays.
    reduction : {'max', 'sum'}
        Reduction over f. See compute_cdp.
    ax : matplotlib axis, optional
        Plot on this axis. Creates a new figure if None.
    log_scale : bool
        Use a log y-axis.
    alpha_range : tuple (low, high), optional
        Restrict the x-axis range.
    title : str, optional
        Plot title.
    normalize : bool
        Divide each profile by its α=0 value (or max if α=0 absent),
        so all algorithms share the same scale for shape comparison.

    Returns
    -------
    ax : matplotlib axis
    """
    if isinstance(estimates, dict):
        items = [(label, Sx, alpha) for label, (Sx, alpha) in estimates.items()]
    else:
        items = list(estimates)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (label, Sx, alpha) in enumerate(items):
        alpha_u, profile = compute_cdp(Sx, alpha, reduction=reduction)

        if normalize:
            idx_zero = np.argmin(np.abs(alpha_u))
            scale = profile[idx_zero] if abs(alpha_u[idx_zero]) < 1e-6 else profile.max()
            profile = profile / scale if scale > 0 else profile

        color = colors[i % len(colors)]

        # Stem for sparse outputs (e.g. S3CA), line for dense
        if len(alpha_u) < 500:
            markerline, stemlines, baseline = ax.stem(
                alpha_u, profile, linefmt=color, markerfmt=color,
                basefmt=' ', label=label,
            )
            markerline.set_markersize(3)
            stemlines.set_linewidth(0.7)
            stemlines.set_alpha(0.6)
        else:
            ax.plot(alpha_u, profile, color=color, label=label,
                    linewidth=0.7, alpha=0.8)

    if log_scale:
        ax.set_yscale('log')

    if alpha_range is not None:
        ax.set_xlim(alpha_range)

    ax.set_xlabel('Cycle frequency α')
    ylabel = 'I(α) = max_f |C(f, α)|' if reduction == 'max' else 'Σ_f |C(f, α)|'
    if normalize:
        ylabel += '  (normalized to α=0)'
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    return ax


def plot_cyclic_domain_profile(spectral_coherence, f, alpha, conjugate=False, plot=True):
    """Legacy single-estimator CDP plot. Prefer compute_cdp / plot_cdp for new code."""
    alpha_u, cdp = compute_cdp(spectral_coherence, alpha)

    if not conjugate:
        mask = alpha_u >= 0
        alpha_u = alpha_u[mask]
        cdp = cdp[mask]

    if plot:
        title = "Conjugate" if conjugate else "Non-Conjugate"
        plt.figure(figsize=(7, 3))
        plt.title("Cyclic Domain Profile")
        plt.xlabel(f"{title} Cycle Frequency (Hz)")
        plt.ylabel("Maximum over frequency")
        plt.stem(alpha_u, cdp)

    return alpha_u, cdp


def cyclic_domain_profile_test(func_lambda, Np=64, L=16, snr=10, signal_length=32768, conjugate=False, plot=True, fam=False, tolerance_bins=2, threshold=0.2):
    samples_per_symbol = 10
    cfo = 0.05

    bpsk_signal = create_rect_bpsk_signal(signal_length, samples_per_symbol)
    bpsk_signal_cfo = add_cfo(bpsk_signal, cfo)
    bpsk_signal_noise_cfo = add_awgn_snr(bpsk_signal_cfo, desired_snr=snr)[:signal_length]

    N = len(bpsk_signal_noise_cfo)

    if fam:
        spectral_coherence, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, L=L, conjugate=conjugate, coherence=True)
    else:
        spectral_coherence, f, alpha = func_lambda(bpsk_signal_noise_cfo, Np=Np, conjugate=conjugate, coherence=True)

    cf, cdp = compute_cdp(spectral_coherence, alpha)

    if not conjugate:
        mask = cf >= 0
        cf = cf[mask]
        cdp = cdp[mask]

    # Pad so find_peaks can detect peaks at boundaries
    cdp_padded = np.pad(cdp, 1)
    peak_indices, properties = find_peaks(cdp_padded, height=threshold, distance=3)
    detected_alphas = cf[peak_indices]
    detected_heights = properties['peak_heights']

    f_sym = 1 / samples_per_symbol
    if conjugate:
        expected_alphas = np.arange(-1, 1, f_sym) + 2 * cfo
    else:
        expected_alphas = np.arange(0, 1, f_sym)
    expected_alphas = expected_alphas[(expected_alphas <= cf.max()) & (expected_alphas >= cf.min())]

    tolerance = tolerance_bins / N
    missing_peaks = [e for e in expected_alphas if not np.any(np.abs(detected_alphas - e) <= tolerance)]
    erroneous_peaks = [d for d in detected_alphas if not np.any(np.abs(expected_alphas - d) <= tolerance)]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(cf, cdp, label="CDP")
        plt.axhline(threshold, color='r', linestyle='--', label="Threshold")
        plt.plot(detected_alphas, detected_heights, "x", color='k', markersize=8, label="Detected Peaks")
        for ea in expected_alphas:
            plt.axvline(ea, color='g', linestyle=':', alpha=0.5)
        plt.title("Automated Peak Detection on Cyclic Domain Profile Test")
        plt.xlabel("Cycle Frequency (Hz)")
        plt.legend()
        plt.show()

    passed = len(missing_peaks) == 0 and len(erroneous_peaks) == 0
    return passed, {
        "detected_alphas": detected_alphas,
        "missing_peaks": missing_peaks,
        "erroneous_peaks": erroneous_peaks,
        "threshold_used": threshold,
    }


def _coherence(func_lambda, signal, Np, L, conjugate, fam):
    if fam:
        coherence, f, alpha = func_lambda(signal, Np=Np, L=L, conjugate=conjugate, coherence=True)
    else:
        coherence, f, alpha = func_lambda(signal, Np=Np, conjugate=conjugate, coherence=True)
    return np.abs(np.asarray(coherence)).ravel(), np.asarray(f).ravel(), np.asarray(alpha).ravel()


def signal_identification_test(func_lambda, signals=None, Np=64, L=16, snr=10, signal_length=32768, conjugate=False,
                               fam=False, factor=3.0, rng=None, plot=True):
    """Cycle frequency identification test on the signals of scf_test.signals.

    For each signal, the cyclic domain profile I(alpha) = max_f |C(f, alpha)| of one realization is compared with
    the signal's significant cycle frequencies from theory. The floor is the median of the profile away from them.
    An expected cycle frequency is found when the profile within two bins of it exceeds `factor` times the floor;
    a spurious peak is any other peak above that level, outside 1/Np of an expected cycle frequency (its cycle
    leakage), outside the edges |alpha| > 1 - 1/Np and, for the non-conjugate SCD, outside |alpha| < 2/Np (the
    alpha = 0 ridge). Cycle frequencies
    outside the estimator's output range (e.g. the Fast-SC's alpha <= fs/2) are not counted.

    Returns
    -------
    results : dict
        For each signal: expected, found and missing cycle frequencies, spurious peaks, the floor, and the
        contrast (weakest found feature over the floor).
    """
    from .signals import SIGNALS, expected_cycle_frequencies, make_signal

    if signals is None:
        signals = [name for name in SIGNALS if not conjugate or SIGNALS[name].conjugate]
    rng = np.random.default_rng() if rng is None else rng
    results = {}

    if plot:
        fig, axs = plt.subplots(len(signals), 1, figsize=(8, 1.3 * len(signals)), sharex=True, squeeze=False)

    for row, name in enumerate(signals):
        x, _ = make_signal(name, signal_length, snr=snr, rng=rng)
        coherence, _, alpha = _coherence(func_lambda, x, Np, L, conjugate, fam)
        cf, cdp = compute_cdp(coherence, alpha)
        if not conjugate:
            keep = cf > 0
            cf, cdp = cf[keep], cdp[keep]

        expected = expected_cycle_frequencies(name, conjugate)
        expected = expected[(expected >= cf.min()) & (expected <= cf.max())]
        tolerance = 2 / signal_length
        guard = np.zeros(len(cf), bool)
        for e in expected:
            guard |= np.abs(cf - e) <= 1 / Np
        guard |= np.abs(cf) > 1 - 1 / Np
        if not conjugate:
            guard |= cf < 2 / Np
        floor = np.median(cdp[~guard]) if np.any(~guard) else np.median(cdp)
        level = factor * floor

        heights = np.array([cdp[np.abs(cf - e) <= tolerance].max() if np.any(np.abs(cf - e) <= tolerance) else 0
                            for e in expected])
        found = expected[heights > level]
        peaks, _ = find_peaks(np.pad(cdp, 1), height=level, distance=3)
        peaks = peaks - 1
        spurious = cf[peaks[~guard[peaks]]] if len(peaks) else np.array([])

        results[name] = {
            "expected": expected,
            "found": found,
            "missing": np.setdiff1d(expected, found),
            "spurious": spurious,
            "floor": floor,
            "contrast": heights[heights > level].min() / floor if np.any(heights > level) else 0.0,
        }

        if plot:
            ax = axs[row, 0]
            ax.plot(cf, cdp, color="0.3", linewidth=0.6)
            for e in expected:
                ax.axvline(e, color="tab:green", linewidth=0.6, alpha=0.5, zorder=0)
            for s in spurious:
                ax.axvline(s, color="tab:red", linewidth=0.8, linestyle=":")
            ax.axhline(level, color="tab:orange", linewidth=0.6, linestyle="--")
            ax.set_ylim(0, None)
            ax.text(0.01, 0.75, f"{SIGNALS[name].label}: {len(found)}/{len(expected)} found, "
                                f"{len(spurious)} spurious", transform=ax.transAxes, fontsize=8)

    if plot:
        axs[-1, 0].set_xlabel(f"{'Conjugate' if conjugate else 'Non-conjugate'} cycle frequency")
        fig.tight_layout()
        plt.show()

    return results


def signal_robustness_test(func_lambda, signals=None, Np=64, L=16, N=4096, snr=(0, -5, -10), trials=500,
                           conjugate=False, fam=False, rng=None, plot=True):
    """Robustness test (ROC) on the signals of scf_test.signals.

    Statistic: max_f |C(f, alpha_0)| on the slice alpha_0 of each signal (roc_alpha_nc or roc_alpha_c in the
    catalogue). H1: the signal in AWGN at the given SNR; H0: the same noise without the signal. Signals without
    a feature of the requested kind are skipped.

    Returns
    -------
    auc : dict
        For each signal, the AUC at each SNR.
    statistics : dict
        For each signal, an array (SNR, [present, absent], trial) of the statistic.
    """
    from .signals import SIGNALS, make_signal

    if signals is None:
        signals = list(SIGNALS)
    signals = [name for name in signals
               if (SIGNALS[name].roc_alpha_c if conjugate else SIGNALS[name].roc_alpha_nc) is not None]
    rng = np.random.default_rng() if rng is None else rng
    auc, statistics = {}, {}

    for name in signals:
        alpha_0 = SIGNALS[name].roc_alpha_c if conjugate else SIGNALS[name].roc_alpha_nc
        values = np.zeros((len(snr), 2, trials))
        slice_index = None
        for i, s in enumerate(snr):
            for t in range(trials):
                noisy, clean = make_signal(name, N, snr=s, rng=rng)
                for k, x in enumerate((noisy, noisy - clean)):
                    coherence, _, alpha = _coherence(func_lambda, x, Np, L, conjugate, fam)
                    if slice_index is None:
                        resolution = np.median(np.diff(np.unique(np.round(alpha, 10))))
                        slice_index = np.flatnonzero(np.abs(alpha - alpha_0) <= resolution / 2)
                    values[i, k, t] = coherence[slice_index].max()
        statistics[name] = values
        labels = np.concatenate([np.ones(trials), np.zeros(trials)])
        auc[name] = np.array([metrics.roc_auc_score(labels, values[i].ravel()) for i in range(len(snr))])

    if plot and auc:
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(auc)), 3))
        width = 0.8 / len(snr)
        for i, s in enumerate(snr):
            ax.bar(np.arange(len(auc)) + (i - (len(snr) - 1) / 2) * width, [auc[n][i] for n in auc], width,
                   label=f"{s} dB")
        ax.axhline(0.5, color="0.5", linewidth=0.6, linestyle=":")
        ax.set_xticks(np.arange(len(auc)), [SIGNALS[n].label for n in auc], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"{'Conjugate' if conjugate else 'Non-conjugate'} robustness test, N = {N}, Np = {Np}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()

    return auc, statistics

