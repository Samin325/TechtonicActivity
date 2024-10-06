import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import spectrogram

stream = read('xa.s12.00.mhz.1972-01-04HR00_evid00049.mseed')
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
weighted_power_threshold = np.percentile(weighted_power, 90)  # Top 10% of weighted power

# identify frequency ranges with significant weighted power to determine the low and high cutoff frequencies
significant_freqs = frequencies[weighted_power > weighted_power_threshold]
low_cutoff = significant_freqs[0] if len(significant_freqs) > 0 else 0.1  # Use first valid frequency
high_cutoff = significant_freqs[-1] if len(significant_freqs) > 0 else 1.5  # Use last valid frequency

print(f"Suggested low cutoff: {low_cutoff} Hz")
print(f"Suggested high cutoff: {high_cutoff} Hz")


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
plt.show()
