import numpy as np
import pandas as pd
from obspy import read
import os
import glob
from skopt import gp_minimize
from skopt.space import Real
import pickle
import os
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from helpers import calculate_noise_level
import bisect

from helpers import implement_STA_LTA, score_function, show_sound_plot, split_files_into_buckets

mars_training_folder_path = './space_apps_2024_seismic_detection/data/mars/training/data'
mars_training_mseed_files = glob.glob(
    os.path.join(mars_training_folder_path, '*.mseed'))

mars_test_folder_path = './space_apps_2024_seismic_detection/data/mars/test/data'
mars_test_mseed_files = glob.glob(
    os.path.join(mars_test_folder_path, '*.mseed'))

mars_catalog_csv_file = './space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv'


def categorize(trace):
    noise = calculate_noise_level(trace.data)
    index = bisect.bisect(mars_training_buckets_max, noise)
    return index


def objective_function(params):
    sta_len, lta_len, thr_on, thr_off = params
    total_f1 = 0
    total_acc = 0

    for quak_file in bucket_data:
        detected_triggers = implement_STA_LTA(
            quak_file['trace'], sta_len, lta_len, thr_on, thr_off)

        # show_sound_plot(quak_file['trace'], detected_triggers)
        f1, accuracy = score_function(
            mars_true_triggers.loc[mars_true_triggers['filename'] == quak_file['filename']]['time_rel(sec)'], detected_triggers)
        total_f1 += f1
        total_acc += accuracy

    return -total_f1


def find_f1(params, dats, showplot=False):
    sta_len, lta_len, thr_on, thr_off = params
    total_f1 = 0
    total_acc = 0

    for quak_file in dats:
        detected_triggers = implement_STA_LTA(
            quak_file['trace'], sta_len, lta_len, thr_on, thr_off)

        if showplot:
            show_sound_plot(quak_file['trace'], detected_triggers)
        f1, accuracy = score_function(
            mars_true_triggers.loc[mars_true_triggers['filename'] == quak_file['filename']]['time_rel(sec)'], detected_triggers)
        total_f1 += f1
        total_acc += accuracy

    return -total_f1


def filter_bp(v_original: np.ndarray, fs, fl, fh, ntaps=513) -> np.ndarray:
    wl = fl / (fs / 2)
    wh = fh / (fs / 2)
    bp_coeff = signal.firwin(
        ntaps, [wl, wh], pass_zero="bandpass", window="hamming")
    return signal.lfilter(bp_coeff, 1, v_original)


if __name__ == '__main__':
    all_mars_training_data = []
    for file_path in mars_training_mseed_files:
        filename = os.path.splitext(file_path)[0]
        st = read(file_path)
        tr = st[0]
        all_mars_training_data.append({'trace': tr, 'filename': filename})
    mars_true_triggers = pd.read_csv(mars_catalog_csv_file)

    buckets, mars_training_buckets_max = split_files_into_buckets(
        all_mars_training_data, num_buckets=5)
    print(f'Bucket boundaries: {mars_training_buckets_max}')

    search_space = [
        Real(50, 200, name='sta_len'),
        Real(300, 900, name='lta_len'),
        Real(2, 4, name='thr_on'),
        Real(0.5, 2.0, name='thr_off')
    ]
    best_params_for_bucket = {}
    for bucket_num, buck in buckets.items():
        bucket_data = buck

        result = gp_minimize(objective_function, search_space, n_calls=70)
        print(f"Bucket {bucket_num}:")
        best_params = result.x
        best_params_for_bucket[bucket_num] = best_params
        print(f"Best parameters for Bayse optimizer: {best_params}")
        print(f"Average f1: {-objective_function(best_params)/len(buck)}")
        print()

    with open('best_params_for_bucket_mars.pkl', 'wb') as f:
        pickle.dump(best_params_for_bucket, f)

    with open('mars_training_buckets_max.pkl', 'wb') as f:
        pickle.dump(mars_training_buckets_max, f)

    fl = 0.3
    fh = 1.1
    test_lunar_folder = './space_apps_2024_seismic_detection/data/lunar/test/data/'
    test_lunar_csvs = glob.glob(os.path.join(
        test_lunar_folder, '**', '*.mseed'))
    cwd = os.getcwd()
    with open(cwd+'/sta-lta/best_params_for_bucket_mars.pkl', 'rb') as f:
        best_params_for_bucket = pickle.load(f)
    with open(cwd+'/sta-lta/mars_training_buckets_max.pkl', 'rb') as f:
        mars_training_buckets_max = pickle.load(f)

    output = []

    for file_path in test_lunar_csvs:
        filename = os.path.splitext(file_path)[0]
        st = read(file_path)
        tr = st[0]
        v_original = tr.data
        v_filtered = filter_bp(v_original, tr.stats.sampling_rate, fl, fh)
        tr.data = v_filtered

        index = categorize(tr)
        params = best_params_for_bucket[index + 1]
        detections = implement_STA_LTA(
            tr, params[0], params[1], params[2], params[3])
        for rel in detections:
            output.append({
                'filename': filename.split('/')[-1]+'.mseed',
                'relative_time': rel,
            })
        output_df = pd.DataFrame(output)
        output_df.to_csv(
            './sta-lta/mars_catalogue_reltiveTime.csv', index=False)
