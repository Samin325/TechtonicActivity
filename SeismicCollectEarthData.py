import csv
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import glob

# Function to clear the directory
def clear_directory(directory_path):
    files = glob.glob(f"{directory_path}*")
    for file in files:
        os.remove(file)
    print(f"Directory '{directory_path}' has been cleared.")

# Function to get seismic data from IRIS FDSNWS
def get_seismic_data(network, station, location, channel, starttime, endtime, file_name, directory_path):
    base_url = "https://service.iris.edu/fdsnws/dataselect/1/query"
    params = {
        "net": network,
        "sta": station,
        "loc": location,
        "cha": channel,
        "start": starttime,
        "end": endtime,
        "format": "miniseed"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Save the MiniSEED file
        with open(f"{directory_path}{file_name}.mseed", 'wb') as f:
            f.write(response.content)
        print(f"Seismic data saved as {file_name}.mseed")
    else:
        print(f"Failed to retrieve data for {file_name}: {response.status_code}")

# Function to process earthquake CSV and download 24-hour MiniSEED data
def process_earthquake_csv(csv_file, directory_path, data_catalog_file):
    # Clear the directory before starting
    clear_directory(directory_path)
    
    # Read the CSV file containing earthquake catalog
    df = pd.read_csv(csv_file)
    
    # Open the new catalog file for writing
    with open(data_catalog__file, mode='w', newline='') as catalog:
        catalog_writer = csv.writer(catalog)
        # Write the header
        catalog_writer.writerow(['filename', 'time_abs(%Y-%m-%dT%H:%M:%S.%f)', 'time_rel(sec)', 'id'])

        # Loop through each earthquake event
        for index, row in df.iterrows():
            # Convert epoch time to datetime
            event_time = datetime.utcfromtimestamp(row['time'] / 1000)

            # Make it a random time between 1 and 24 hours                 
            r_hour = random.randint(1, 24)
            start_time = event_time - timedelta(hours=r_hour)  # Random start time before event
            end_time = event_time + timedelta(hours=24 - r_hour)  # End time after event

            # Define file name based on event time and location
            file_name = f"quake_{row['latitude']}_{row['longitude']}_{event_time.strftime('%Y%m%dT%H%M%S')}"

            # Specify seismic station and network info (you can change these based on your needs)
            network = "IU"   # Example network (Global Seismic Network)
            station = "ANMO"  # Example station
            location = "00"   # Location code
            channel = "BHZ"   # Channel code

            # Get 24-hour MiniSEED data
            get_seismic_data(network, station, location, channel, start_time.isoformat(), end_time.isoformat(), file_name, directory_path)
            
            file_name += ".mseed"
            
            # Calculate time_abs and time_rel
            time_abs = event_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
            time_rel = (event_time - start_time).total_seconds()  # Relative time is 0 at the event time

            # Create an identifier for the event
            event_id = f"quake_{index + 1}"

            # Write the event information to the new catalog
            catalog_writer.writerow([file_name, time_abs, time_rel, event_id])
            
    print(f"New catalog created: {data_catalog__file}")
# Example usage: process the earthquake catalog CSV and download 24-hour MiniSEED files
data_directory = './data/earth/mseed/'
data_catalog__file = './data/earth/new_earth_earthquake_catalog.csv'
process_earthquake_csv('./data/earth/earthquake_catalog.csv', data_directory, data_catalog__file)