low_cutoff = 0.26526  # determined low cutoff frequency
high_cutoff = 0.976929  # determined high cutoff frequency

import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import spectrogram, medfilt

stream = read('xa.s12.00.mhz.1972-01-04HR00_evid00049.mseed')
trace = stream[0]
velocity = trace.data
sampling_rate = trace.stats.sampling_rate

# generate the spectrogram
frequencies, times, Sxx = spectrogram(velocity, fs=sampling_rate, nperseg=1024)

# filter out frequencies not within the cutoff range
valid_freq_indices = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
filtered_Sxx = Sxx[valid_freq_indices, :]

# apply a median filter to reduce noise without affecting event timing (adjustable kernel size)
total_power_in_band = np.sum(filtered_Sxx, axis=0)
total_power_filtered = medfilt(total_power_in_band, kernel_size=11)

# 95th percentile power threshold
power_threshold = np.percentile(total_power_filtered, 95)

# identify time intervals where total power exceeds the threshold
activity_intervals = total_power_filtered > power_threshold

# collect the start and end times of significant activity intervals
start_times = []
end_times = []
in_activity = False
min_duration = 60  # adjustable minimum duration of significant activity in seconds
current_start = None

for i in range(len(activity_intervals)):
    if activity_intervals[i] and not in_activity:
        # Start of a new activity interval
        current_start = times[i]
        in_activity = True
    elif not activity_intervals[i] and in_activity:
        # End of the current activity interval
        current_end = times[i]
        in_activity = False
        # Only register the activity if it lasts longer than the minimum duration
        if current_end - current_start >= min_duration:
            start_times.append(current_start)
            end_times.append(current_end)

# Print the start and end times of detected activity
for start, end in zip(start_times, end_times):
    print(f"Seismic activity detected from {start:.2f} seconds to {end:.2f} seconds")

# Plot total power in the band over time and highlight activity intervals
plt.figure(figsize=(12, 6))
plt.plot(times, total_power_filtered, label='Filtered Total Power in Band (Median Filtered)')
plt.axhline(power_threshold, color='red', linestyle='--', label='Power Threshold')
for start, end in zip(start_times, end_times):
    plt.axvspan(start, end, color='green', alpha=0.3, label='Detected Activity' if start == start_times[0] else "")
plt.title('Detected Seismic Activity Based on Frequency Filtering (Median Filtered)')
plt.xlabel('Time (seconds)')
plt.ylabel('Filtered Total Power in Band ((m/s)^2/Hz)')
plt.legend()
plt.show()
