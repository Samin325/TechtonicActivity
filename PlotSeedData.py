# scroll_plotter.py
import os
import matplotlib.pyplot as plt
import pandas as pd
from SeedReader import read_mseed

class ScrollPlotter:
    def __init__(self, path_to_data, catalog_file, plots_per_page=4):
        self.current_index = 0
        self.plots_per_page = plots_per_page

        # Read the CSV file containing earthquake event information
        self.event_data = pd.read_csv(catalog_file)

        # Filter mseed files based on the CSV file
        self.mseed_files = [os.path.join(path_to_data, f) for f in self.event_data['filename'].tolist()]

        # Create the figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 8))  # 2 rows, 2 cols, adjust as needed
        self.axes = self.axes.flatten()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # self.plot_files()

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
        try:
            # Call the read_mseed function from mseed_reader.py
            tr_times, tr_data, arrival_time = read_mseed(file_name, self.event_data)

            # Plot trace on the provided axis
            ax.plot(tr_times, tr_data)

            # Mark the earthquake event with a red vertical line
            ax.axvline(x=arrival_time, color='red', linestyle='--', label=f"Earthquake Event ({arrival_time:.2f}s)")

            # Make the plot pretty
            ax.set_xlim([min(tr_times), max(tr_times)])
            ax.set_ylabel('Velocity (m/s)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{os.path.basename(file_name)}', fontweight='bold')
            ax.legend(loc='upper right')

        except Exception as e:
            print(f"Error plotting file {file_name}: {e}")

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
if __name__ == "__main__":
    data_directory = './data/earth/mseed'
    catalog_file = './data/earth/new_earth_earthquake_catalog.csv'
    plotter = ScrollPlotter(data_directory, catalog_file)
    plotter.plot_files()
    plt.show()
