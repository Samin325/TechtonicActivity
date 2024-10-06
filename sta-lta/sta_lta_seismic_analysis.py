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
import bisect
from skopt import gp_minimize
from skopt.space import Real

from helpers import implement_STA_LTA, score_function, show_sound_plot, calculate_noise_level, split_files_into_buckets

lunar_training_folder_path = './space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
lunar_training_mseed_files = glob.glob(os.path.join(lunar_training_folder_path, '*.mseed'))

lunar_test_folder_path = './space_apps_2024_seismic_detection/data/lunar/test/data/S16_GradeA'
lunar_test_mseed_files = glob.glob(os.path.join(lunar_test_folder_path, '*.mseed'))

lunar_catalog_csv_file = './space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'


def objective_function(params):
    sta_len, lta_len, thr_on, thr_off = params
    total_f1 = 0
    total_acc = 0

    for quak_file in bucket_data:
        detected_triggers = implement_STA_LTA(quak_file['trace'], sta_len, lta_len, thr_on, thr_off)

        # show_sound_plot(quak_file['trace'], detected_triggers)
        f1, accuracy = score_function(lunar_true_triggers.loc[lunar_true_triggers['filename'] == quak_file['filename']]['time_rel(sec)'], detected_triggers)
        total_f1 += f1
        total_acc += accuracy

    return -total_f1

def find_f1(params, dats, showplot=False):
    sta_len, lta_len, thr_on, thr_off = params
    total_f1 = 0
    total_acc = 0

    for quak_file in dats:
        detected_triggers = implement_STA_LTA(quak_file['trace'], sta_len, lta_len, thr_on, thr_off)

        if showplot:
            show_sound_plot(quak_file['trace'], detected_triggers)
        f1, accuracy = score_function(lunar_true_triggers.loc[lunar_true_triggers['filename'] == quak_file['filename']]['time_rel(sec)'], detected_triggers)
        total_f1 += f1
        total_acc += accuracy

    return -total_f1

def categorize(trace):
    noise = calculate_noise_level(trace.data)
    index = bisect.bisect(lunar_training_buckets_max, noise)
    return index

if __name__ == '__main__':
    all_lunar_training_data = []
    for file_path in lunar_training_mseed_files:
        filename = file_path.split('/')[-1].removesuffix('.mseed')
        st = read(file_path)
        tr = st[0]
        all_lunar_training_data.append({'trace': tr, 'filename': filename})
    lunar_true_triggers = pd.read_csv(lunar_catalog_csv_file)

    buckets, lunar_training_buckets_max = split_files_into_buckets(all_lunar_training_data, num_buckets=5)
    print(f'Bucket boundaries: {lunar_training_buckets_max}')

    search_space = [
        Real(50, 200, name='sta_len'),
        Real(300, 900, name='lta_len'),
        Real(2, 4, name='thr_on'),
        Real(0.5, 2.0, name='thr_off')
    ]
    best_params_for_bucket={}
    for bucket_num, buck in buckets.items():
        bucket_data = buck

        result = gp_minimize(objective_function, search_space, n_calls=100)
        print(f"Bucket {bucket_num}:")
        best_params = result.x
        best_params_for_bucket[bucket_num] = best_params
        print(f"Best parameters for Bayse optimizer: {best_params}")
        print(f"Average f1: {-objective_function(best_params)/len(buck)}")
        print()    

    for quak_file in all_lunar_training_data:
        index = categorize(quak_file['trace'])
        params = best_params_for_bucket[index + 1]
        print(quak_file['filename'])
        f1 = find_f1(params, [quak_file], True)
        print(index)

