import requests
import csv

# Function to query the USGS earthquake catalog
def query_earthquake_catalog(start_time, end_time, min_magnitude=5.0, limit=100):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude,
        "limit": limit
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        events = []
        for event in data["features"]:
            properties = event["properties"]
            events.append({
                "time": properties["time"],  # Time in milliseconds since epoch
                "latitude": event["geometry"]["coordinates"][1],
                "longitude": event["geometry"]["coordinates"][0],
                "depth": event["geometry"]["coordinates"][2],
                "magnitude": properties["mag"],
            })
        return events
    else:
        print(f"Error: {response.status_code}")
        return []

# Function to store earthquake catalog data as a CSV
def store_raw_catalog_as_csv(earthquakes, directory_path, file_name='raw_earthquake_catalog.csv'):
    # Define the CSV columns
    columns = ["time", "latitude", "longitude", "depth", "magnitude"]
    
    # Write to the CSV file
    with open(f"{directory_path}{file_name}", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        
        # Write the header
        writer.writeheader()
        
        # Write the earthquake data
        writer.writerows(earthquakes)
    
    print(f"Catalog data saved to {directory_path}{file_name}")
    

if __name__ == "__main__":
    earthquakes = query_earthquake_catalog("2022-01-01", "2022-12-31", min_magnitude=5, limit=100)

    file_name = "raw_earth_catalog"
    directory_path = "./data/earth/"
    store_raw_catalog_as_csv(earthquakes, directory_path, file_name)
