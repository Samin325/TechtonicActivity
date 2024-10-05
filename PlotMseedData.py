import os
from matplotlib import pyplot as plt
from obspy import read
import math

class ScrollPlotter:
    def __init__(self, directory, plots_per_page=4):
        self.mseed_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mseed')]
        self.current_index = 0
        self.plots_per_page = plots_per_page

        # Create the figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 8))  # 2 rows, 2 cols, adjust as needed
        self.axes = self.axes.flatten()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.plot_files()

    def plot_files(self):
        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Plot the current set of files
        for i in range(self.plots_per_page):
            if self.current_index + i < len(self.mseed_files):
                file_name = self.mseed_files[self.current_index + i]
                print(f"Processing file: {file_name}")
                self.plot_mseed(file_name, self.axes[i])

        # Redraw the figure
        plt.tight_layout()
        plt.draw()

    def plot_mseed(self, file_name, ax):
        st = read(file_name)
        tr = st.traces[0]
        tr_times = tr.times()
        tr_data = tr.data

        # Start time of trace
        starttime = tr.stats.starttime.datetime
        print(f"Start Time: {starttime}")

        # Plot trace on the provided axis
        ax.plot(tr_times, tr_data)

        # Make the plot pretty
        ax.set_xlim([min(tr_times), max(tr_times)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{os.path.basename(file_name)}', fontweight='bold')

    def on_key(self, event):
        # Navigate forward or backward with left/right arrow keys
        if event.key == 'right':
            if self.current_index + self.plots_per_page < len(self.mseed_files):
                self.current_index += self.plots_per_page
        elif event.key == 'left':
            if self.current_index - self.plots_per_page >= 0:
                self.current_index -= self.plots_per_page
        self.plot_files()

# Example usage
data_directory = './data/earth/mseed'
plotter = ScrollPlotter(data_directory)
plt.show()
