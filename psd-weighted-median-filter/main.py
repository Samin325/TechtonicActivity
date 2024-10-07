import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import spectrogram, medfilt
import re

def resolveTimes(stream):
  trace = stream[0]
  velocity = trace.data
  sampling_rate = trace.stats.sampling_rate

  # generate spectrogram data
  frequencies, times, Sxx = spectrogram(velocity, fs=sampling_rate, nperseg=1024)

  # power spectral density
  power_sqrtHz = np.sqrt(Sxx)
  # 95th percentile power threshold (basic power highpass)
  power_threshold = np.percentile(power_sqrtHz, 95)

  # count how long the power remains above the threshold at each frequency
  duration_above_threshold = np.zeros(len(frequencies))
  for i in range(len(frequencies)):
      above_threshold = power_sqrtHz[i, :] > power_threshold  # Boolean array where power is above the threshold
      duration_above_threshold[i] = np.sum(above_threshold) * (times[1] - times[0])  # Duration in seconds

  # weight the total power against how long it stays above the threshold
  weighted_power = np.sum(Sxx, axis=1) * duration_above_threshold
  # 90th percentile weighted power threshold
  weighted_power_threshold = np.percentile(weighted_power, 95)  # Top 10% of weighted power

  # identify frequency ranges with significant weighted power to determine the low and high cutoff frequencies
  significant_freqs = frequencies[weighted_power > weighted_power_threshold]
  low_cutoff = significant_freqs[0] if len(significant_freqs) > 0 else 0.1  # Use first valid frequency
  high_cutoff = significant_freqs[-1] if len(significant_freqs) > 0 else 1.5  # Use last valid frequency

  print(f"Suggested low cutoff: {low_cutoff} Hz")
  print(f"Suggested high cutoff: {high_cutoff} Hz")

  '''
  plt.figure(figsize=(10, 6))

  # plot duration above threshold
  plt.subplot(2, 1, 1)
  plt.plot(frequencies, duration_above_threshold, label='Duration Above Threshold (s)')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Duration (seconds)')
  plt.title('Duration of High-Power Oscillations')
  plt.legend()

  # plot weighted power
  plt.subplot(2, 1, 2)
  plt.plot(frequencies, weighted_power, label='Weighted Power by Duration')
  plt.axhline(weighted_power_threshold, color='r', linestyle='--', label=f'Threshold: {weighted_power_threshold:.2e}')
  plt.fill_between(frequencies, weighted_power, where=(weighted_power > weighted_power_threshold), color='green', alpha=0.5)
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Weighted Power')
  plt.title('Weighted Power by Duration of High-Power Oscillations')
  plt.legend()

  plt.tight_layout()
  #plt.show()
  '''
  # filter out frequencies not within the cutoff range
  valid_freq_indices = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
  filtered_Sxx = Sxx[valid_freq_indices, :]

  # apply a median filter to reduce noise without affecting event timing (adjustable kernel size)
  total_power_in_band = np.sum(filtered_Sxx, axis=0)
  total_power_filtered = medfilt(total_power_in_band, kernel_size=11)

  # 95th percentile power threshold
  power_threshold = np.percentile(total_power_filtered, 98)

  # identify time intervals where total power exceeds the threshold
  activity_intervals = total_power_filtered > power_threshold

  # collect the start and end times of significant activity intervals
  start_times = []
  end_times = []
  in_activity = False
  min_duration = 160  # adjustable minimum duration of significant activity in seconds
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

  # Plot total power in the band over time and highlight activity intervals
  '''
  plt.figure(figsize=(12, 6))
  plt.plot(times, total_power_filtered, label='Filtered Total Power in Band (Median Filtered)')
  plt.axhline(power_threshold, color='red', linestyle='--', label='Power Threshold')
  for start, end in zip(start_times, end_times):
      plt.axvspan(start, end, color='green', alpha=0.3, label='Detected Activity' if start == start_times[0] else "")
  plt.title('Detected Seismic Activity Based on Frequency Filtering (Median Filtered)')
  plt.xlabel('Time (seconds)')
  plt.ylabel('Filtered Total Power in Band ((m/s)^2/Hz)')
  plt.legend()
  #plt.show()
  '''
  return zip(start_times, end_times)


dataDir = "data/lunar/test/data/S16_GradeB"
resultFile = open("apollo12_catalog_S16_GradeB.csv", "a")

for filename in os.listdir(dataDir):
  print(filename)
  if "mseed" in filename:
    stream = read(dataDir + '/' + filename)
    data = resolveTimes(stream)
    for start, end in data:
        evidTag = re.findall(r"evid[0-9]{5}", filename)[0]
        resultFile.write(f"{filename},{start:.2f},{evidTag}\n")
        print(f"Seismic activity detected from {start:.2f} seconds to {end:.2f} seconds")
