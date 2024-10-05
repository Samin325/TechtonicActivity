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
def process_earthquake_csv(csv_file, directory_path):
    # Clear the directory before starting
    clear_directory(directory_path)
    
    # Read the CSV file containing earthquake catalog
    df = pd.read_csv(csv_file)
    
    # Loop through each earthquake event
    for index, row in df.iterrows():
        # Convert epoch time to datetime
        event_time = datetime.utcfromtimestamp(row['time'] / 1000)

        # make it a random time between 1 and 24 hours                 
        r_hour = random.randint(1,24)
        
        start_time = event_time - timedelta(hours=r_hour)  # 12 hours before the event
        end_time = event_time + timedelta(hours=24-r_hour)  # 12 hours after the event
        
        # Define file name based on event time and location
        file_name = f"quake_{row['latitude']}_{row['longitude']}_{event_time.strftime('%Y%m%dT%H%M%S')}"

        # Specify seismic station and network info (you can change these based on your needs)
        network = "IU"   # Example network (Global Seismic Network)
        station = "ANMO"  # Example station
        location = "00"   # Location code
        channel = "BHZ"   # Channel code

        # Get 24-hour MiniSEED data
        get_seismic_data(network, station, location, channel, start_time.isoformat(), end_time.isoformat(), file_name, directory_path)

# Example usage: process the earthquake catalog CSV and download 24-hour MiniSEED files
data_directory = './data/earth/mseed/'
process_earthquake_csv('./data/earth/earthquake_catalog.csv', data_directory)