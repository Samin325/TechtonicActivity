# bp_filter.py

# imports
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

# read data, extract vectors/constants
datapath = "space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"
file = "xa.s12.00.mhz.1970-07-20HR00_evid00011.csv"
T = pd.read_csv(datapath + file)

t = T['time_rel(sec)'].values
v = T['velocity(m/s)'].values
fs = 1 / (t[1] - t[0])

# define moon-quakes as 0.3-1.2 Hz band
fl = 0.3
fh = 1.2
wl = fl / (fs / 2)
wh = fh / (fs / 2)

# create bandpass and bandstop filter coefficients
bp_coeff = signal.firwin(513, [wl, wh], pass_zero="bandpass", window="hamming")
bs_coeff = signal.firwin(513, [wl, wh], pass_zero="bandstop", window="hamming")

# isolate the in-band and noise (out-of-band) elements of the signal
v_filt = signal.lfilter(bp_coeff, 1, v)
v_noise = signal.lfilter(bs_coeff, 1, v)

# create a spectrogram
plt.figure(figsize=(10, 6))
plt.specgram(v_filt, NFFT=1000, Fs=fs, noverlap=200, cmap='turbo')
plt.colorbar(label='Intensity [dB]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Filtered Signal')
plt.ylim(0, 2)  # Adjust the frequency limits if necessary
plt.show()
