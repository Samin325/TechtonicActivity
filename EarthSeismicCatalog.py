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
                "place": properties["place"]
            })
        return events
    else:
        print(f"Error: {response.status_code}")
        return []

