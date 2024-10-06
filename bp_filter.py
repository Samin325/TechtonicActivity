# bp_filter.py

# imports
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# read data, extract vectors/constants
DATAPATH = "space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"
FILE = "xa.s12.00.mhz.1970-07-20HR00_evid00011.csv"


def get_time_velocity(filename: str) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(filename)
    return table['time_rel(sec)'].values, table['velocity(m/s)'].values


def filter_bp(v_original: np.ndarray, fs, fl, fh, ntaps=513) -> np.ndarray:
    wl = fl / (fs / 2)
    wh = fh / (fs / 2)
    bp_coeff = signal.firwin(ntaps, [wl, wh], pass_zero="bandpass", window="hamming")
    return signal.lfilter(bp_coeff, 1, v_original)


def filter_bs(v_original: np.ndarray, fs, fl, fh, ntaps=513) -> np.ndarray:
    wl = fl / (fs / 2)
    wh = fh / (fs / 2)
    bs_coeff = signal.firwin(ntaps, [wl, wh], pass_zero="bandstop", window="hamming")
    return signal.lfilter(bs_coeff, 1, v_original)


def plot_spectrogram(x: np.ndarray, noverlap: int, nfft: int, fs) -> None:
    plt.figure(figsize=(10, 6))
    plt.specgram(x,
                 noverlap=noverlap,
                 NFFT=nfft,
                 Fs=fs,
                 cmap='turbo')
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram of Filtered Signal')
    plt.ylim(0, 2)  # Adjust the frequency limits if necessary
    plt.show()


if __name__ == "__main__":
    t, v = get_time_velocity(DATAPATH + FILE)
    Fs = 1 / (t[1] - t[0])

    # define moon-quakes as 0.3-1.1 Hz band
    fl = 0.3
    fh = 1.1

    v_filt = filter_bp(v, Fs, fl, fh)
    v_noise = filter_bs(v, Fs, fl, fh)
    # TODO export v_filt and v_noise

    plot_spectrogram(v_filt, 200, 1000, Fs)
    plot_spectrogram(v_noise, 200, 1000, Fs)
