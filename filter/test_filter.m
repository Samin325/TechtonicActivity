% read data and extract vectors/constants
datapath = "space_apps_2024_seismic_detection/" ...
        + "data/lunar/training/data/S12_GradeA/";
file = "xa.s12.00.mhz.1970-07-20HR00_evid00011.csv";
T = readtable(datapath + file);
t = T.time_rel_sec_;
v = T.velocity_m_s_;
fs = 1/(t(2) - t(1));

% define earthquakes as 0.3-1.2 Hz band
fl = 0.3;
fh = 1.2;
wl = fl/(fs/2);
wh = fh/(fs/2);

% create bandpass and bandstop filter coefficients
window = hamming(513);
bp_coeff = fir1(512, [wl wh], "bandpass", window);
bs_coeff = fir1(512, [wl wh], "stop", window);

% isolate the in-band and noise (out-of-band) elements of the signal
v_filt = filter(bp_coeff, 1, v);
v_noise = filter(bs_coeff, 1, v);

% spectrogram(v, window, 200, 1000, fs, "yaxis");
% spectrogram(v_filt, window, 200, 1000, fs, "yaxis");
% spectrogram(v_noise, window, 200, 1000, fs, "yaxis");
