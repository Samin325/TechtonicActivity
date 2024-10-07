# TechtonicActivity
"No comet"

This repository contains 4 different algorithms to predict moonquakes and marsquakes given miniSEED files over a 24-hour period.

Each of the algorithms can be found in the following folders:

- Short-term average/long-term average with noise-level categorization - sta-lta
- Power spectral density analysis - psd-weighted-median-filter
- Convolutional neural network trained on miniseed files - cnn-mseed
- Convolutional neural network trained on images of waveforms - cnn-image

We believe PSD analysis (algorithm 2) to be the most accurate, based on results from training data.
Please use the catalogs from the psd-weighted-median-filter directory for scoring.

Complementary band-pass and band-stop filters are used to separate the quake-prone and noise components of the signals.
These filtered signal components were passed to the other algorithms to reduce the impact of unreliable raw input.

We supplemented the provided dataset by NASA with earthquake data for training our machine learning models. 
We used USGS API to get a catalog of earthquakes that occured. 
We used that data to query earthquake data from SAGE to get miniSEED files.

Models were trained on GCP instances. To set up the environment, run script.sh in the directory that the model will be running in.
