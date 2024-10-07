# TechtonicActivity
"No comet"

This repository contains 4 different algorithms to predict moonquakes and marsquakes given miniSEED files over a 24-hour period.

Each of the algorithms can be found in the following folders:

- Short-term average/long-term average with noise-level categorization - sta-lta
- Power spectral density analysis - psd-weighted-median-filter
- Convolutional neural network trained on miniseed files - cnn-mseed
- Convolutional neural network trained on images of waveforms - cnn-image



Models were trained on GCP instances. To set up the environment, run script.sh in the directory that the model will be running in.
