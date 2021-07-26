import numpy as np
from scipy import signal


def smooth(data):
    return np.apply_along_axis(lambda d: signal.medfilt(d, kernel_size=21), axis=1, arr=data)


def angle_shift(ang):
    if np.mean(ang) > 2.8:
        return ang - np.pi
    if np.mean(ang) < -2.8:
        return ang + np.pi
    return ang


def norm_emg(data, **kwargs):
    emg_std = np.std(data, axis=1)
    emg_mean = np.mean(data, axis=1)
    return (data - emg_mean[:, None]) / emg_std[:, None]


def bp_filter(data, high_band=7, low_band=400, sfreq=2000, filt_ord=2, causal=True, **kwargs):
    data = norm_emg(data)
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)
    # create bandpass filter for EMG
    b, a = signal.butter(filt_ord, [high_band, low_band], btype='bandpass', output='ba')
    # process EMG signal: filter EMG
    return signal.lfilter(b,a, data, axis=1) if causal else signal.filtfilt(b,a, data, axis=1)


