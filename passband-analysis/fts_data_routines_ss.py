import numpy as np
import math
import matplotlib.pyplot as plt
from so3g.hk import load_range
# from sotodlib.flags import get_glitch_flags
from sotodlib.tod_ops.flags import get_glitch_flags, get_trending_flags
from sotodlib.tod_ops.detrend import detrend_tod
import latrt_testing.fft_ops as fft_ops
import latrt_testing.demodulation as demod
import os
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter

#### relaading latrt-testing demod since making edits to it ###
import importlib
import latrt_testing.demodulation as demod
importlib.reload(demod)

print("LOADING NEW fts_data_routines")

HK_KEY = 'satp1.fts-uchicago-so.feeds.position.pos'

def find_time(timestamps, time):
    '''finds closest time index in array timestamps to the inputted time'''
    return (np.abs(timestamps - time)).argmin()


# useful function for analyzing timestreams
def time_zoom(aman, t_min, t_max):
    # returns indices useful for getting a time window
    time = aman.timestamps - aman.timestamps[0]
    inds = np.where((time >= t_min) & (time <= t_max))[0]
    return inds

def psd_snrs_hwp(aman, chop_freq=8, nperseg=2**10, apply_trend_cuts=True):
    """Take out trends"""
    # # get the 'good' dets with lots of power at 8Hz
    # if (plot):
    #     plt.figure()

    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)
    
    # good_dets = []
    psd_snrs = []
    psd_snrs.clear()

    peak_freqs = []
    peak_freqs.clear()
    # good_dets.clear()
    
    if apply_trend_cuts:
        trend_cuts = aman.flags.has_cuts(['trends'])
    for i in range(0, aman.dets.count):
        if apply_trend_cuts:
            if aman.dets.vals[i] in trend_cuts:
                continue
                
        # Find the maximum frequency since there is some slight offset
        window=2 # Seeing where exactly freq max is 8GHz +/- 2GHz
        # 1. Create a boolean mask for the frequency window
        mask = (freqs >= chop_freq - window) & (freqs <= chop_freq + window)
        
        # 2. Find the index of the peak within that window
        peak_idx = np.argmax(Pxx[i][mask])
        
        # 3. Convert back to the index in the original array
        peak_idx_global = np.where(mask)[0][peak_idx]
        
        peak_freq = freqs[peak_idx_global]
        peak_value = Pxx[i][peak_idx_global]
        
        psd_snr = Pxx[i, peak_idx_global] / (Pxx[i, peak_idx_global + 10])
        psd_snrs.append(psd_snr)
        peak_freqs.append(peak_freq)

    return peak_freqs, psd_snrs

def get_good_dets(aman, Pxx, freqs, power_threshold=1000, plot=False,
                  chop_freq=8, apply_trend_cuts=True):

    """Take out trends"""
    # get the 'good' dets with lots of power at 8Hz
    if (plot):
        plt.figure()

    good_dets = []
    if apply_trend_cuts:
        trend_cuts = aman.flags.has_cuts(['trends'])
    for i in range(0, aman.dets.count):
        if apply_trend_cuts:
            if aman.dets.vals[i] in trend_cuts:
                continue

        index_of_chophz = np.where(freqs <= chop_freq)[0][-1] # restricts to desired chop freq
        if Pxx[i, index_of_chophz] > (Pxx[i, index_of_chophz + 10] * (power_threshold)):
            # # also check where max occurs because some dets might have a sizable peak at both 8Hz and 32Hz with 8Hz peak dominating
            # # mask off data below 5Hz since PSD peaks around 0Hz
            # mask = np.where(freqs >= 5)
            # # find max after applying that mask
            # maxind_masked = np.argmax(Pxx[i][mask])
            # maxind = mask[0][maxind_masked]
            # # freq where max occurs
            # if (freqs[maxind] <= chop_freq+2) & (freqs[maxind] >= chop_freq-2):
            #    good_dets.append(i)
            good_dets.append(i)
            if (plot):
                plt.semilogy(freqs, Pxx[i])

    # see the good dets array
    print("Good dets:", good_dets)
    print(np.shape(np.array(good_dets)))
    
    if (plot):
        # plt.xlim(chop_freq - 2, chop_freq + 2)
        plt.xlim(6, 40)
        plt.axvline(8, ls='--')
        plt.axvline(32, ls='--')
        # plt.grid()
        plt.xlabel('frequency (hz)')
        plt.ylabel('P(f)')
        plt.title(f'power spectra of detectors with good {chop_freq}hz power')
        # plt.ticklabel_format(style='plain', axis='x')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.show()

    good_dets = np.array(good_dets)

    return good_dets
    
def get_good_dets_hwp(aman, Pxx, freqs, power_threshold=1000, plot=False,
                  chop_freq=8, apply_trend_cuts=True):

    """Take out trends"""
    # get the 'good' dets with lots of power at 8Hz
    if (plot):
        plt.figure()

    good_dets = []
    if apply_trend_cuts:
        trend_cuts = aman.flags.has_cuts(['trends'])
    for i in range(0, aman.dets.count):
        if apply_trend_cuts:
            if aman.dets.vals[i] in trend_cuts:
                continue
        # Find the maximum frequency since there is some slight offset
        window=2 # Seeing where exactly freq max is 8GHz +/- 2GHz
        # 1. Create a boolean mask for the frequency window
        mask = (freqs >= chop_freq - window) & (freqs <= chop_freq + window)
        
        # 2. Find the index of the peak within that window
        peak_idx = np.argmax(Pxx[i][mask])
        
        # 3. Convert back to the index in the original array
        peak_idx_global = np.where(mask)[0][peak_idx]
        
        peak_freq = freqs[peak_idx_global]
        peak_value = Pxx[i][peak_idx_global]
        
        if Pxx[i, peak_idx_global] > (Pxx[i, peak_idx_global + 10] * power_threshold):
            good_dets.append(i)
            if (plot):
                plt.semilogy(freqs, Pxx[i])

    # see the good dets array
    print("Good dets:", good_dets)
    print(np.shape(np.array(good_dets)))
    
    if (plot):
        # plt.xlim(chop_freq - 2, chop_freq + 2)
        plt.xlim(6, 40)
        # plt.axvline(8, ls='--')
        # plt.axvline(32, ls='--')
        # plt.grid()
        plt.xlabel('frequency (hz)')
        plt.ylabel('P(f)')
        plt.title(f'power spectra of detectors with good {chop_freq}hz power')
        # plt.ticklabel_format(style='plain', axis='x')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.show()

    good_dets = np.array(good_dets)

    return good_dets

def get_fts_ind_ranges(fts_position_inds):
    time_interval = 200
    ind_ranges = []
    for i, inds in enumerate(fts_position_inds):
        ind_start, ind_end = inds[0], inds[-1]
        if len(inds) > 2:
            ind_start, ind_end = inds[0], inds[-2]
        # If there's only one housekeeping index (happens rarely with >1s
        # integration unless it skips a data point), get the previous one
        # which is further away and integrate in that direction.
        if ind_start == ind_end:
            next_ind = fts_position_inds[i + 1][0]
            prev_ind = fts_position_inds[i - 1][-1]
            if (next_ind - ind_start) > (ind_start - prev_ind):
                ind_end = int(ind_start + time_interval / 2)
            else:
                ind_start = int(ind_end - time_interval / 2)

        # Integrate between these
        ind_range = np.arange(ind_start, ind_end + 1)
        ind_ranges.append(ind_range)
    return ind_ranges


def get_integration_indices_optimized(fts_ind_ranges, glitch_mask):
    total_non_glitch_inds = []
    for ind_range in fts_ind_ranges:
        mask = glitch_mask[np.where((glitch_mask >= ind_range[0]) & (
            glitch_mask <= ind_range[-1]))]
        non_glitch_inds = np.setdiff1d(ind_range, mask)
        total_non_glitch_inds.append(non_glitch_inds)
    return total_non_glitch_inds


def get_integration_indices(fts_position_inds, glitch_mask):
    # So we just need to integrate between all of our times and discount the
    # glitches
    # First we need to make sure that the glitches exist
    time_interval = 200
    total_non_glitch_inds = []
    for i, inds in enumerate(fts_position_inds):
        ind_start, ind_end = inds[0], inds[-1]
        if ind_start == ind_end:
            # get the previous one which is further away and integrate in that
            # direction.
            next_ind = fts_position_inds[i + 1][0]
            prev_ind = fts_position_inds[i - 1][-1]
            if (next_ind - ind_start) > (ind_start - prev_ind):
                ind_end = int(ind_start + time_interval / 2)
            else:
                ind_start = int(ind_end - time_interval / 2)

        # Integrate between these
        ind_range = np.arange(ind_start, ind_end + 1)
        non_glitch_inds = np.setdiff1d(ind_range, glitch_mask)
        total_non_glitch_inds.append(non_glitch_inds)
    return total_non_glitch_inds


def integrate_signal(signal, total_non_glitch_inds):
    return np.array([np.mean(signal[inds]) for inds in total_non_glitch_inds])


def load_fts_range(aman, resolution=.15):
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        data_dir="/so/level2-daq/satp1/hk",
        fields = ['satp1.fts-uchicago-so.feeds.position.pos'])

    max_position = -1 * np.round(np.min(hk_data[HK_KEY][1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position,
        int(2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data[HK_KEY][1]
    # now take out the initial data chunk
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[last_max_index:]
    hk_times = hk_data[HK_KEY][0][last_max_index:]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= .01)[0]
        if len(hk_inds) == 0:
            print(f"no housekeeping data at fts position {pos}. "
                  "Using data from previous position")
            hk_position = hk_mirror_slice[-1]
            hk_time = hk_time_slice[-1]
        else:
            hk_index = hk_inds[0]
            hk_position = hk_mirror_positions[hk_index]
            hk_time = hk_times[hk_index]
        hk_mirror_slice.append(hk_position)
        hk_time_slice.append(hk_time)

    #assert (np.abs(hk_mirror_slice - expected_fts_mirror_positions) <= .01).all()

    aman_fts_position_timeslice = np.array(
        [find_time(aman.timestamps, time) for time in hk_time_slice])
    return aman_fts_position_timeslice, np.array(hk_mirror_slice)


def load_fts_range_bounds(hk_data, aman, resolution=.15, max_position=None):
    # hk_data = load_range(
    #     float(aman.timestamps[0]), float(aman.timestamps[-1]),
    #     data_dir="/so/level2-daq/satp1/hk",
    #     fields = ['satp1.fts-uchicago-so.feeds.position.pos'])

    if max_position is None:
        max_position = -1 * np.round(np.min(hk_data[1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position, int(
            2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data[1]
    # now take out the initial data chunk
    # start slightly after the beginning to account for any weird trends
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][2]
    # start slightly before the end similarly
    first_right_max_index = np.where(
        np.abs(hk_mirror_positions - max_position) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[
        last_max_index: first_right_max_index]
    hk_times = hk_data[0][last_max_index: first_right_max_index]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= .01)[0]
        if len(hk_inds) == 0:
            print(f"no housekeeping data at fts position {pos}. "
                  "Using data from previous position")
            hk_position = hk_mirror_slice[-1]
            hk_time = hk_time_slice[-1]
        else:
            hk_position = hk_mirror_positions[hk_inds][0]
            hk_time = hk_times[hk_inds]
        hk_mirror_slice.append(hk_position)
        hk_time_slice.append(hk_time)

    aman_fts_position_timeslice = [
        [find_time(aman.timestamps, time) for time in s] for s in (
            hk_time_slice)]
    return hk_mirror_slice, aman_fts_position_timeslice


def plot_good_interferograms(aman, good_dets, signal, fts_mirror_positions,
                             figsize=(10, 10)):
    n_bias_groups = np.max(aman.det_info.smurf.bias_group) + 1
    fig, axes = plt.subplots(math.ceil(n_bias_groups / 2), 2, figsize=figsize)
    axes = axes.ravel()
    trend_cuts = aman.flags.has_cuts(['trends'])

    for group in range(n_bias_groups):
        axes[group].grid(True)
        count = np.sum(aman.det_info.smurf.bias_group[good_dets] == group)
        axes[group].set_title(
            "bias group %s, number of 'good' dets = %s" % (group, count))

    print('number of interferograms in bias group -1: %s' % np.sum(
        aman.det_info.smurf.bias_group[good_dets] == -1))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in trend_cuts or np.max(
                signal[i]) > 1:
            continue

        group = aman.det_info.smurf.bias_group[i]
        if (group != -1) and i in good_dets:
            axes[group].plot(fts_mirror_positions, signal[i])
    plt.tight_layout()
    plt.show()


def plot_good_interferograms_bands(aman, good_dets, signal, fts_mirror_positions,
                                   figsize=(10, 10)):
    n_bands= np.max(aman.det_info.smurf.band) + 1
    fig, axes = plt.subplots(math.ceil(n_bands / 2), 2, figsize=figsize)
    axes = axes.ravel()
    trend_cuts = aman.flags.has_cuts(['trends'])

    for group in range(n_bands):
        axes[group].grid(True)
        count = np.sum(aman.det_info.smurf.band[good_dets] == group)
        axes[group].set_title(
            "band %s, number of 'good' dets = %s" % (group, count))

    print('number of interferograms in band -1: %s' % np.sum(
        aman.det_info.smurf.band[good_dets] == -1))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in trend_cuts or np.max(
                signal[i]) > 1:
            continue

        group = aman.det_info.smurf.band[i]
        if (group != -1) and i in good_dets:
            axes[group].plot(fts_mirror_positions, signal[i])
    plt.tight_layout()
    plt.show()


def save_data_v2(aman, band_type, n, fts_mirror_positions, signal, group1_inds, no_group_inds, mean_phases, phaseg1, folder_name,
              band_channel_map, file_suffix):
    # fts_x, fts_y = get_fts_position(aman)
    # get obs_id
    obs_id = aman["obs_info"]["obs_id"]
    
    # get det_names
    det_names = aman.dets.vals
    
    # save the data for loading in from another notebook
    trend_cuts = aman.flags.has_cuts(['trends'])
    data = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
    bands = np.zeros(len(band_channel_map))
    channels = np.zeros(len(band_channel_map))
    for i in range(aman.dets.count):
        band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
        band_channel_id = band_channel_map[(band, channel)]
        if aman.dets.vals[i] in trend_cuts:
            data[:, band_channel_id] = signal[i]
        else:
            data[:, band_channel_id] = signal[i]
        bands[band_channel_id] = band
        channels[band_channel_id] = channel
    filename = f'%s/run%s_{band_type}GHz{file_suffix}_interferograms.npz' % (folder_name, n)
    
    with open(filename, 'wb') as f:
        np.savez(f, data=data, fts_mirror_positions=fts_mirror_positions, group1_inds = group1_inds, no_group_inds = no_group_inds, mean_phases = mean_phases, phaseg1 = phaseg1, 
                 bands=bands, channels=channels, det_names = det_names, obs_id=obs_id)
    print('data saved to location %s' % filename)
    return
    
# def save_data_v2(aman, band_type, n, fts_mirror_positions, signal, group1_inds, no_group_inds, mean_phases, phaseg1, folder_name,
#               band_channel_map):
#     # fts_x, fts_y = get_fts_position(aman)

#     # get obs_id
#     obs_id = aman["obs_info"]["obs_id"]
    
#     # get det_names
#     det_names = aman.dets.vals
    
#     # save the data for loading in from another notebook
#     trend_cuts = aman.flags.has_cuts(['trends'])
#     data = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
#     bands = np.zeros(len(band_channel_map))
#     channels = np.zeros(len(band_channel_map))
#     for i in range(aman.dets.count):
#         band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
#         band_channel_id = band_channel_map[(band, channel)]
#         # print(band, channel)
#         if aman.dets.vals[i] in trend_cuts:
#             # just make this data a bunch of zeros
#             # adjust this to actually save data-- don't trust trends lol
#             data[:, band_channel_id] = signal[i]
#             # data[:, band_channel_id] = np.zeros(len(fts_mirror_positions))
#         else:
#             data[:, band_channel_id] = signal[i]

#         bands[band_channel_id] = band
#         channels[band_channel_id] = channel

#     # filename = f'%s/run%s_{band_type}GHz_interferograms_pW_260527.npz' % (folder_name, n)
#     filename = f'%s/run%s_{band_type}GHz_interferograms_pW_260614.npz' % (folder_name, n)
    
#     with open(filename, 'wb') as f:
#         np.savez(f, data=data, fts_mirror_positions=fts_mirror_positions, group1_inds = group1_inds, no_group_inds = no_group_inds, mean_phases = mean_phases, phaseg1 = phaseg1, 
#                  bands=bands, channels=channels, det_names = det_names, obs_id=obs_id)
#     print('data saved to location %s' % filename)
#     return

def save_data_hwp(aman, band_type, n, fts_mirror_positions, signal, signalQ, signalU, folder_name,
              band_channel_map, file_suffix):
    # fts_x, fts_y = get_fts_position(aman)

    # get obs_id
    obs_id = aman["obs_info"]["obs_id"]
    
    # get det_names
    det_names = aman.dets.vals
    
    # save the data for loading in from another notebook
    trend_cuts = aman.flags.has_cuts(['trends'])
    data = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
    dataQ = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
    dataU = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
    bands = np.zeros(len(band_channel_map))
    channels = np.zeros(len(band_channel_map))
    for i in range(aman.dets.count):
        band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
        band_channel_id = band_channel_map[(band, channel)]
        # print(band, channel)
        if aman.dets.vals[i] in trend_cuts:
            # just make this data a bunch of zeros
            # adjust this to actually save data-- don't trust trends lol
            data[:, band_channel_id] = signal[i]
            dataQ[:, band_channel_id] = signalQ[i]
            dataU[:, band_channel_id] = signalU[i]
            # data[:, band_channel_id] = np.zeros(len(fts_mirror_positions))
        else:
            data[:, band_channel_id] = signal[i]
            dataQ[:, band_channel_id] = signalQ[i]
            dataU[:, band_channel_id] = signalU[i]

        bands[band_channel_id] = band
        channels[band_channel_id] = channel

    filename = f'%s/HWPrun%s_{band_type}GHz_interferograms{file_suffix}.npz' % (folder_name, n)
    with open(filename, 'wb') as f:
        np.savez(f, data=data, dataQ = dataQ, dataU = dataU, fts_mirror_positions=fts_mirror_positions, 
                 bands=bands, channels=channels, det_names = det_names, obs_id=obs_id)
    print('data saved to location %s' % filename)
    return

def save_data_ss(aman, n, fts_mirror_positions, signals, demod_signal_names, demod_phases, num_phase_groups, phase_group, det_phase, snr_raw, det_names, folder_name, band_channel_map, obs_id):
    # fts_x, fts_y = get_fts_position(aman)

    # save the data for loading in from another notebook
    trend_cuts = aman.flags.has_cuts(['trends'])

    # Data to hold the interferograms with the different phase demodulations
    data = [np.zeros((len(fts_mirror_positions), len(band_channel_map))) for _ in range(num_phase_groups)]

    bands = np.zeros(len(band_channel_map))
    channels = np.zeros(len(band_channel_map))
    
    for i in range(aman.dets.count):
        band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
        band_channel_id = band_channel_map[(band, channel)]
        # print(band, channel)

        for j in range(num_phase_groups):
            data[j][:, band_channel_id] = signals[j][i]
        else:
            data[j][:, band_channel_id] = signals[j][i]

        bands[band_channel_id] = band
        channels[band_channel_id] = channel

    save_dict = {f'data{j}': data[j] for j in range(num_phase_groups)}
    save_dict.update(dict(fts_mirror_positions=fts_mirror_positions, bands=bands, channels=channels, demod_signal_names  = demod_signal_names, demod_phases = demod_phases, num_phase_groups = num_phase_groups, phase_group = phase_group, det_phase = det_phase, snr_raw = snr_raw, det_names  = det_names, obs_id=obs_id))
    
    filename = '%s/run_%s_interferograms.npz' % (folder_name, n)
    with open(filename, 'wb') as f:
        np.savez(f, **save_dict)
    print('data saved to location %s' % filename)
    return

def check_chopper_signal_hwp(aman, power_threshold=100, chop_freq=8, peak_freq = None,
                         nperseg=2**10, return_good_aman=False):
    print(f"Length of chop = {aman.timestamps[-1] - aman.timestamps[0]}")
    # get_trending_flags(aman, t_piece=50)
    # detrend_tod(aman)
    # print("Detrended aman")
    # Pxx, freqs = fft_ops.psd(aman, nperseg=nperseg)

    print("Chop freq: ", chop_freq)
    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)

    print("Plot PSDs")
    plt.figure()
    for psd in Pxx[::10]:
        plt.plot(freqs, psd)
    plt.semilogy()
    plt.show()
    
    good_dets = get_good_dets_hwp(aman, Pxx, freqs, plot=False,
                              power_threshold=power_threshold,
                              chop_freq=chop_freq, apply_trend_cuts=False)
    
    # print("Good dets:", good_dets)
    print(f"Power threshold: {power_threshold}, Number of good dets = {len(good_dets)}.")
    plt.plot(freqs, Pxx[good_dets].T, alpha=0.1)
    if peak_freq != None:
        plt.axvline(peak_freq, label=f'{np.round(peak_freq, 2)}')
    plt.yscale('log')
    # plt.axvline(13, ls='--', color="black")
    # plt.axvline(2.8, ls='--', color="black")
    # plt.axvline(chop_freq, ls='--', color="black")
    plt.xlim(0, chop_freq+20)
    # plt.ylim(1e-8, 1e-1)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.legend()
    plt.show()

    # # removing below steps because not necessary for HWP modulated data
    good_aman = aman.restrict('dets', aman.dets.vals[good_dets], in_place=False) # used to be phase_fit_aman
    
    # mid_rel_t = round((phase_fit_aman.timestamps[-1] - phase_fit_aman.timestamps[0]) / 2.)
    
    # # print("Middle relative time for demod.fit_phase:", mid_rel_t)
    # # print("Calling demodulation.py from latrt-testing")
    # phase_to_use, phases = demod.fit_phase(phase_fit_aman, mid_rel_t, plot=True,
    #                                        threshold=0.5, index_limit=180,
    #                                        freq=chop_freq)
    # if np.std(phases) > .3:
    #     print('Phase fitting standard deviation is slightly high, check hist')
    #     plt.hist(phases, bins=20)
    #     plt.xlim(0, 6.28)
    #     plt.xlabel('phase')
    #     plt.ylabel('counts')
    #     plt.grid()
    #     plt.show()

    # demod.demod_single_sine(phase_fit_aman, phase_to_use, lp_fc=0.5,
    #                         freq=chop_freq)

    # plt.plot(phase_fit_aman.timestamps - phase_fit_aman.timestamps[0],
    #          phase_fit_aman.demod_signal.T, alpha=0.3)
    
    # [plt.axhline(m, alpha=0.1, ls="--", color=f"C{i}") for i, m in enumerate(
    #     np.median(phase_fit_aman.demod_signal, axis=1))]
    # #plt.ylim(-0.02, 0.4)
    # #plt.xlim(20, 20 + (5 / 32))
    # plt.show()
    # if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
    #     print_num_in_each_band(phase_fit_aman)
    # if not return_good_aman:
    #     return
        
    return good_aman


def check_chopper_signal(aman, power_threshold=100, chop_freq=8,
                         nperseg=2**10, return_good_aman=False, full_phase_interfs = True, savefigurepath = 0):
    print(f"Length of chop = {aman.timestamps[-1] - aman.timestamps[0]}")
    # get_trending_flags(aman, t_piece=50)
    # detrend_tod(aman)
    # print("Detrended aman")
    # Pxx, freqs = fft_ops.psd(aman, nperseg=nperseg)

    print("Chop freq: ", chop_freq)
    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)

    # print("Plot PSDs")
    # plt.figure()
    # for psd in Pxx[::10]:
    #     plt.plot(freqs, psd)
    # plt.semilogy()
    # plt.show()
    
    good_dets = get_good_dets(aman, Pxx, freqs, plot=False,
                              power_threshold=power_threshold,
                              chop_freq=chop_freq, apply_trend_cuts=False)
    
    # print("Good dets:", good_dets)
    print(f"Power threshold: {power_threshold}, Number of good dets = {len(good_dets)}.")

    plt.figure()
    plt.title(f'Threshold = {power_threshold}, {len(good_dets)} good dets')
    plt.plot(freqs, Pxx[good_dets].T, alpha=0.1)
    plt.yscale('log')
    # plt.axvline(13, ls='--', color="black")
    # plt.axvline(2.8, ls='--', color="black")
    plt.axvline(chop_freq, ls='--', color="black")
    plt.xlim(0, chop_freq+20)
    plt.ylim(1e-8, 1e-1)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    if (savefigurepath):
        plt.savefig(savefigurepath)
    plt.show()

    if (full_phase_interfs):
        phase_fit_aman = aman.restrict('dets', aman.dets.vals[good_dets], in_place=False)
        mid_rel_t = round((phase_fit_aman.timestamps[-1] - phase_fit_aman.timestamps[0]) / 2.)
        # print("Middle relative time for demod.fit_phase:", mid_rel_t)
        # print("Calling demodulation.py from latrt-testing")
        phase_to_use, phases = demod.fit_phase(phase_fit_aman, mid_rel_t, plot=True,
                                               threshold=0.5, index_limit=180,
                                               freq=chop_freq)
        if np.std(phases) > .3:
            print('Phase fitting standard deviation is slightly high, check hist')
            plt.hist(phases, bins=20)
            plt.xlim(0, 6.28)
            plt.xlabel('phase')
            plt.ylabel('counts')
            plt.grid()
            plt.show()
    
        demod.demod_single_sine(phase_fit_aman, phase_to_use, lp_fc=0.5,
                                freq=chop_freq)
    
        plt.plot(phase_fit_aman.timestamps - phase_fit_aman.timestamps[0],
                 phase_fit_aman.demod_signal.T, alpha=0.3)
        [plt.axhline(m, alpha=0.1, ls="--", color=f"C{i}") for i, m in enumerate(
            np.median(phase_fit_aman.demod_signal, axis=1))]
        #plt.ylim(-0.02, 0.4)
        #plt.xlim(20, 20 + (5 / 32))
        plt.show()
    else:
        phase_fit_aman = aman.restrict('dets', aman.dets.vals[good_dets], in_place=False)
        
    # if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
        # print_num_in_each_band(phase_fit_aman)
    if not return_good_aman:
        return
    return phase_fit_aman

def get_middle_psd(aman, middle_ind=None, window_length=12000,
                   nperseg=2**8):
    """Get PSD using a chunk of aman in the middle."""
    if middle_ind is None:
        middle_ind = aman.timestamps.shape[0] // 2
    psd_aman = aman.restrict('samps', (middle_ind - window_length // 2,
                                       middle_ind + window_length // 2),
                             in_place=False)
    Pxx, freqs = fft_ops.psd(psd_aman, nperseg=nperseg)
    return Pxx, freqs

def process_run_ufm(aman, fts_pos, folder_name, band_channel_map, phase_to_use,
                    middle_relative_time=2000, threshold=.1, index_limit=160,
                    plot=False, resolution=.1, nperseg=(2 ** 9),
                    demod_lp_fc=0.5, chop_freq=8, max_position=None,
                    run_num=0):
    assert os.path.exists(folder_name)
    get_trending_flags(aman)
    detrend_tod(aman)

    fts_mirror_positions, fts_time_ranges = load_fts_range_bounds(fts_pos, aman, resolution=0.15, max_position=None)
    
    # get the glitches.
    _ = get_glitch_flags(aman, hp_fc=1.0, buffer=20, overwrite=True, n_sig=50)
    mask = aman.flags.glitches.mask()

    # get the 'good' dets with lots of power at 8Hz
    # Pxx, freqs = fft_ops.psd(aman, nperseg=nperseg)
    # Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)
    # for power_threshold in [100, 10]:
    #     print('using power threshold of %s:' %power_threshold)
    #     good_dets = get_good_dets(aman, Pxx, freqs, plot=plot,
    #                               power_threshold=power_threshold,
    #                               chop_freq=chop_freq)
    #     if len(good_dets) > 80:
    #         break
    # # print('number of detectors with higher power in 8hz = %s' %len(good_dets))
    
    # if len(good_dets) > 10:
    #     # now fit the phase with the 'good' detectors
    #     phase_fit_aman = aman.restrict('dets', aman.dets.vals[good_dets], in_place=False)
    #     phase_to_use, phases = demod.fit_phase(phase_fit_aman, middle_relative_time, plot=plot,
    #                                            threshold=threshold, index_limit=index_limit,
    #                                            freq=chop_freq)
    #     if np.std(phases) > .3:
    #         print('Phase fitting standard deviation is slightly high, check hist')
    #         plt.hist(phases)
    #         plt.xlabel('phase')
    #         plt.ylabel('counts')
    #         plt.grid()
    #         plt.show()

    #     if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
    #         print_num_in_each_band(phase_fit_aman)

    # demodulate with the fitted phase
    demod.demod_single_sine(aman, phase_to_use, lp_fc=demod_lp_fc, freq=chop_freq)
    print("single sine demodulated")
    
    # else:
    #     print('not enough good detectors found to fit phase. demodulating with a sine + cosine')
    #     demod.demod_sine(aman, freq=chop_freq, lp_fc=demod_lp_fc)

    print("getting integrated signal...")
    # # get the integrated signal
    # fts_mirror_positions, fts_time_ranges  = load_fts_range_bounds(fts_pos, 
    #     aman, resolution=resolution, max_position=max_position)
    interferograms = []
    fts_ind_ranges = get_fts_ind_ranges(fts_time_ranges)
    for i in tqdm(range(len(mask))):
        # integrate around any glitches in the data.
        total_non_glitch_inds = get_integration_indices_optimized(
            fts_ind_ranges, np.where(mask[i])[0])
        integrated_signal = integrate_signal(
            aman.demod_signal[i], total_non_glitch_inds)
        interferograms.append(integrated_signal)
    interferograms = np.array(interferograms)

    if plot:
        if len(good_dets) > 0:
            if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
                plot_good_interferograms(aman, good_dets, interferograms,
                                        fts_mirror_positions)
            else:
                plot_good_interferograms_bands(aman, good_dets, interferograms,
                                            fts_mirror_positions)


        else:
            if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
                plot_good_interferograms(aman, list(range(aman.dets.count)),
                                        interferograms, fts_mirror_positions)

            else:
                plot_good_interferograms_bands(aman, list(range(aman.dets.count)),
                                            interferograms, fts_mirror_positions)


    # save this data along with bias group number, dets, and XY position to another notebook
    save_data(aman, run_num, fts_mirror_positions, interferograms,
              folder_name, band_channel_map, int(aman.timestamps[0]))
    return interferograms

def process_run_ufm_v2(aman, band_type, fts_pos, folder_name, band_channel_map, group1_inds, no_group_inds, mean_phases, file_suffix,
                    plot=False, resolution=.15, nperseg=(2 ** 9),
                    demod_lp_fc=0.5, chop_freq=32., max_position=None,
                    run_num=0,save=True, fts_resolution  = 0.15, single_sine_demod = True):
    """
    aman should also be preprocessed, and good dets should have been identified.
    phase_to_use = mean phase of phase group 1

    Inputs
    band_type: int(90) or (150)
    phase_to_use: mean of phase
    """
    assert os.path.exists(folder_name)

    fts_mirror_positions, fts_time_ranges = load_fts_range_bounds(fts_pos, aman, resolution=fts_resolution, max_position=None)

    if (single_sine_demod):
        # calculate phase_to_use (which is the mean of the phases that are in group 1 -the main phase group)
        phase_g1 = np.mean(np.array(mean_phases)[group1_inds])
        # phase_ng = np.mean(np.array(mean_phases)[no_group_inds])
    
        # demodulate with the fitted phase
        demod.demod_single_sine(aman, phase_g1, lp_fc=demod_lp_fc, freq=chop_freq)
        print("single sine demodulated")
        
        # else:
        #     print('not enough good detectors found to fit phase. demodulating with a sine + cosine')
        #     demod.demod_sine(aman, freq=chop_freq, lp_fc=demod_lp_fc)
    else:
        phase_g1 = np.nan
        # demodulate - phaseless
        demod.demod_sine(aman, freq=chop_freq, lp_fc=demod_lp_fc)
        print("Phaseless demod")

    # get the glitches.
    _ = get_glitch_flags(aman, hp_fc=1.0, buffer=20, overwrite=True, n_sig=50)
    mask = aman.flags.glitches.mask()
    
    print("getting integrated signal...")
    # # get the integrated signal
    # fts_mirror_positions, fts_time_ranges  = load_fts_range_bounds(fts_pos, 
    #     aman, resolution=resolution, max_position=max_position)
    interferograms = []
    fts_ind_ranges = get_fts_ind_ranges(fts_time_ranges)
    for i in tqdm(range(len(mask))):
        # integrate around any glitches in the data.
        total_non_glitch_inds = get_integration_indices_optimized(fts_ind_ranges, np.where(mask[i])[0])
        integrated_signal = integrate_signal(aman.demod_signal[i], total_non_glitch_inds)
        interferograms.append(integrated_signal)
    interferograms = np.array(interferograms)

    # if plot:
    #     if len(good_dets) > 0:
    #         if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
    #             plot_good_interferograms(aman, good_dets, interferograms,
    #                                     fts_mirror_positions)
    #         else:
    #             plot_good_interferograms_bands(aman, good_dets, interferograms,
    #                                         fts_mirror_positions)


    #     else:
    #         if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
    #             plot_good_interferograms(aman, list(range(aman.dets.count)),
    #                                     interferograms, fts_mirror_positions)

    #         else:
    #             plot_good_interferograms_bands(aman, list(range(aman.dets.count)),
    #                                         interferograms, fts_mirror_positions)


    if (save):
        # save this data along with bias group number, dets, and XY position to another notebook
        save_data_v2(aman, band_type, run_num, fts_mirror_positions, interferograms, group1_inds, no_group_inds, mean_phases, phase_g1, folder_name, band_channel_map, file_suffix)
    return interferograms

def process_run_ufm_hwp(aman, band_type, fts_pos, folder_name, band_channel_map, file_suffix, 
                    plot=False, resolution=.15, nperseg=(2 ** 9),
                    demod_lp_fc=0.5, max_position=None,
                    run_num=0, save=True):
    """
    aman should also be preprocessed, and good dets should have been identified.
    phase_to_use = mean phase of phase group 1

    Inputs
    band_type: int(90) or (150)
    phase_to_use: mean of phase
    """
    assert os.path.exists(folder_name)

    fts_mirror_positions, fts_time_ranges = load_fts_range_bounds(fts_pos, aman, resolution=0.15, max_position=None)
    
    # # calculate phase_to_use (which is the mean of the phases that are in group 1 -the main phase group)
    # phase_g1 = np.mean(np.array(mean_phases)[group1_inds])
    # # phase_ng = np.mean(np.array(mean_phases)[no_group_inds])

    # # demodulate with the fitted phase
    # demod.demod_single_sine(aman, phase_g1, lp_fc=demod_lp_fc, freq=chop_freq)
    # print("single sine demodulated")
    
    # else:
    #     print('not enough good detectors found to fit phase. demodulating with a sine + cosine')
    #     demod.demod_sine(aman, freq=chop_freq, lp_fc=demod_lp_fc)

    # get the glitches.
    _ = get_glitch_flags(aman, hp_fc=1.0, buffer=20, overwrite=True, n_sig=50)
    mask = aman.flags.glitches.mask()
    
    print("getting integrated signal...")
    # # # get the integrated signal
    # fts_mirror_positions, fts_time_ranges  = load_fts_range_bounds(fts_pos, 
    #     aman, resolution=resolution, max_position=max_position)
    interferograms = []
    interferogramsQ = []
    interferogramsU = []
    fts_ind_ranges = get_fts_ind_ranges(fts_time_ranges)
    for i in tqdm(range(len(mask))):
        # integrate around any glitches in the data.
        total_non_glitch_inds = get_integration_indices_optimized(fts_ind_ranges, np.where(mask[i])[0])
        integrated_signal = integrate_signal(aman.dsT[i], total_non_glitch_inds)
        integrated_signalQ = integrate_signal(aman.demodQ[i], total_non_glitch_inds)
        integrated_signalU = integrate_signal(aman.demodU[i], total_non_glitch_inds)
        interferograms.append(integrated_signal)
        interferogramsQ.append(integrated_signalQ)
        interferogramsU.append(integrated_signalU)
    interferograms = np.array(interferograms)
    interferogramsQ = np.array(interferogramsQ)
    interferogramsU = np.array(interferogramsU)

    # if plot:
    #     if len(good_dets) > 0:
    #         if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
    #             plot_good_interferograms(aman, good_dets, interferograms,
    #                                     fts_mirror_positions)
    #         else:
    #             plot_good_interferograms_bands(aman, good_dets, interferograms,
    #                                         fts_mirror_positions)


    #     else:
    #         if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
    #             plot_good_interferograms(aman, list(range(aman.dets.count)),
    #                                     interferograms, fts_mirror_positions)

    #         else:
    #             plot_good_interferograms_bands(aman, list(range(aman.dets.count)),
    #                                         interferograms, fts_mirror_positions)


    if save == True:
        # save this data along with bias group number, dets, and XY position to another notebook
        save_data_hwp(aman, band_type, run_num, fts_mirror_positions, interferograms, interferogramsQ, interferogramsU,
                  folder_name, band_channel_map, file_suffix)
    return interferograms, interferogramsQ, interferogramsU
    
def get_interferograms(aman, demod_signal_names, fts_time_ranges):
    '''
    ideally a function to take in an aman and return interferograms, but also incorporates phase groupings and those subsequent demodulations
    '''
    # get the glitches.
    _ = get_glitch_flags(aman, hp_fc=1.0, buffer=20, overwrite=True, n_sig=50)
    mask = aman.flags.glitches.mask()

    interferograms = [[] for _ in range(len(demod_signal_names))] # store interferograms for each demodulation version
    
    fts_ind_ranges = get_fts_ind_ranges(fts_time_ranges)
    
    for i in tqdm(range(len(mask))):
        # integrate around any glitches in the data.
        total_non_glitch_inds = get_integration_indices_optimized(fts_ind_ranges, np.where(mask[i])[0])

        # Below iterates over all the possible phase/phaseless demodulations
        for j, demod_signal_name in enumerate(demod_signal_names):
            aman_signal_i = getattr(aman, demod_signal_name)[i]
            integrated_signal = integrate_signal(aman_signal_i, total_non_glitch_inds)
            interferograms[j].append(integrated_signal)
            
    interferograms = np.array(interferograms)
    return interferograms
    
# def get_fts_position(aman):
#     # get the XY stage position
#     hk_data = load_range(
#         float(aman.timestamps[0]), float(aman.timestamps[-1]),
#         config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')
#     assert np.around(np.std(hk_data['xy_stage_x'][1]), 2) == 0
#     assert np.around(np.std(hk_data['xy_stage_y'][1]), 2) == 0

#     fts_x, fts_y = np.around(np.mean(hk_data['xy_stage_x'][1]), 1), np.round(
#         np.mean(hk_data['xy_stage_y'][1]), 1)
#     return fts_x, fts_y
