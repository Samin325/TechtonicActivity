import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import spectrogram, welch
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset, recursive_sta_lta
import os
import zipfile
import glob
import random


TOLERANCE = 100 # in seconds

def implement_STA_LTA(this_trace, sta_len, lta_len, thr_on, thr_off):
    time_rel = this_trace.times()
    velocity = this_trace.data

    sampling_rate = this_trace.stats.sampling_rate
    try:
        cft = classic_sta_lta(velocity, int(sta_len * sampling_rate), int(lta_len * sampling_rate))
        on_off = np.array(trigger_onset(cft, thr_on, thr_off))
        detected_triggers = [time_rel[triggers[0]] for triggers in on_off if len(triggers) > 0]
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

    if len(on_off) == 0:
        return []

    # print(f"Detected triggers: {detected_triggers}")
    return detected_triggers


def match_triggers(true_triggers, detected_triggers):
    matches = []
    used_detected = set()

    for true_trigger in true_triggers:
        for i, detected_trigger in enumerate(detected_triggers):
            if i not in used_detected and abs(detected_trigger - true_trigger) <= TOLERANCE:
                matches.append((true_trigger, detected_trigger))
                used_detected.add(i)
                break

    return matches

def score_function(true_triggers, detected_triggers):
    # Match detected triggers to true triggers
    matches = match_triggers(true_triggers, detected_triggers)

    # Calculate precision, recall, and F1-score
    num_true = len(true_triggers)
    num_detected = len(detected_triggers)
    num_matches = len(matches)

    precision = num_matches / num_detected if num_detected > 0 else 0
    recall = num_matches / num_true if num_true > 0 else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    total_elements = num_true + num_detected - num_matches
    accuracy = num_matches / total_elements if total_elements > 0 else 0
    return f1_score, accuracy



def calculate_noise_level(velocity):
    return np.std(velocity)

def split_files_into_buckets(all_files, num_buckets):
    noise_levels = [(file['filename'], calculate_noise_level(file['trace'].data), file['trace']) for file in all_files]
    noise_levels.sort(key=lambda x: x[1])
    noise_values = [x[1] for x in noise_levels]

    quantiles = np.linspace(0, 1, num_buckets + 1)
    bucket_boundaries = np.quantile(noise_values, quantiles)

    lunar_training_buckets_max = bucket_boundaries[1:-1]
    buckets = {i+1: [] for i in range(num_buckets)}

    for filename, noise_level, trace in noise_levels:
        for i in range(num_buckets):
            if bucket_boundaries[i] <= noise_level <= bucket_boundaries[i+1]:
                buckets[i+1].append({'filename':filename, 'trace':trace})
                break

    return buckets, lunar_training_buckets_max


# # Bayse optimizer
# from skopt import gp_minimize
# from skopt.space import Real
# search_space = [
#     Real(50, 200, name='sta_len'),
#     Real(300, 900, name='lta_len'),
#     Real(2, 4, name='thr_on'),
#     Real(0.5, 2.0, name='thr_off')
# ]

# bucket_data = all_lunar_training_data
# result = gp_minimize(objective_function, search_space, n_calls=100)

# best_params = result.x
# print(f"Best parameters for Bayse optimizer: {best_params}")
# print(f"Average f1: {-objective_function(best_params)/len(all_lunar_training_data)}")

# # Gradient optimizer
# from scipy.optimize import minimize
# initial_guess = [100.0, 6000.0, 3.0, 1.0]

# bounds = [(50, 200), (300, 900), (2, 4), (0.5, 2.0)]

# result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

# best_params = result.x
# print(f"Best parameters for Gradient optimizer: {best_params}")
# print(f"Average f1: {-objective_function(best_params)/len(all_lunar_training_data)}")

def show_sound_plot(trace, x_values=None):
    time_rel = trace.times()
    velocity = trace.data
    fig, ax = plt.subplots(1,1,figsize=(10,3))
    ax.plot(time_rel, velocity)
    ax.set_xlim([min(time_rel),max(time_rel)])

    if x_values is not None:
        if isinstance(x_values, list):
            for x in x_values:
                ax.axvline(x=x, color='red', linestyle='--', label=f'x = {x}')
        else:
            ax.axvline(x=x_values, color='red', linestyle='--', label=f'x = {x_values}')
    plt.show()




