# mseed_reader.py
import os
from obspy import read
from datetime import datetime

def read_mseed(file_name, event_data = None):
    """
    Read a MiniSEED file and extract the trace data and event information.
    
    Parameters:
    file_name: Path to the MiniSEED file.
    event_data: DataFrame containing earthquake event information.

    Returns:
    tuple: Tuple containing the trace times, trace data, and the relative event time.
    """
    # Read the MiniSEED file
    st = read(file_name)
    tr = st.traces[0]
    tr_times = tr.times()
    tr_data = tr.data

    if event_data:
        # Start time of trace
        starttime = tr.stats.starttime.datetime        
        
        # Get the earthquake event time from the CSV for the current file
        file_base_name = os.path.basename(file_name)
        event_info = event_data[event_data['filename'] == file_base_name]

        if event_info.empty:
            raise ValueError(f"No event info found for file {file_base_name}")

        # Extract absolute event time
        time_abs_str = event_info['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].values[0]
        time_abs = datetime.strptime(time_abs_str, '%Y-%m-%dT%H:%M:%S.%f')

        # Calculate the relative time of the earthquake event with respect to the start of the trace
        arrival_time = (time_abs - starttime).total_seconds()

        return tr_times, tr_data, arrival_time
    else:
        return tr_times, tr_data
