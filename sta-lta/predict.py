import pickle
import os
import bisect
import glob
import pandas as pd
from obspy import read
from helpers import calculate_noise_level
from sta_lta_seismic_analysis import find_f1
from helpers import implement_STA_LTA
import scipy.signal as signal
import numpy as np


def categorize(trace):
    noise = calculate_noise_level(trace.data)
    index = bisect.bisect(lunar_training_buckets_max, noise)
    return index


def filter_bp(v_original: np.ndarray, fs, fl, fh, ntaps=513) -> np.ndarray:
    wl = fl / (fs / 2)
    wh = fh / (fs / 2)
    bp_coeff = signal.firwin(
        ntaps, [wl, wh], pass_zero="bandpass", window="hamming")
    return signal.lfilter(bp_coeff, 1, v_original)


if __name__ == '__main__':
    fl = 0.3
    fh = 1.1
    test_lunar_folder = './space_apps_2024_seismic_detection/data/lunar/test/data/'
    test_lunar_csvs = glob.glob(os.path.join(
        test_lunar_folder, '**', '*.mseed'))
    cwd = os.getcwd()
    with open(cwd+'/sta-lta/best_params_for_bucket_lunar.pkl', 'rb') as f:
        best_params_for_bucket = pickle.load(f)
    with open(cwd+'/sta-lta/lunar_training_buckets_max.pkl', 'rb') as f:
        lunar_training_buckets_max = pickle.load(f)

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
            './sta-lta/lunar_catalogue_reltiveTime.csv', index=False)
