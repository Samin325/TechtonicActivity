import EarthSeismicCatalog
import CollectEarthData
import os

def main():
    # directory names
    mseed_data_directory = "./data/earth/training/data/mseed/"
    catalog_directory = "./data/earth/"
    raw_catalog_file = "raw_earthquake_catalog.csv"
    earthquake_catalog_file = "earthquake_catalog.csv"
    
    os.makedirs(mseed_data_directory, exist_ok=True) # create directory if not exist
    
    # IRIS param
    iris_param = {
        "network" : "IU",   # Example network
        "station" : "ANMO",  # Example station
        "location" : "00",   # Location code
        "channel" : "BHZ"   # Channel code
    }
    
    # get raw data
    raw_start_time = "2022-01-01"
    raw_end_time = "2022-12-31"
    min_magnitude = 5.0
    limit = 300
    
    earthquakes = EarthSeismicCatalog.query_earthquake_catalog(raw_start_time, raw_end_time, min_magnitude, limit)
    EarthSeismicCatalog.store_raw_catalog_as_csv(earthquakes, catalog_directory, raw_catalog_file) # raw catalog file
    
    CollectEarthData.process_earthquake_csv(catalog_directory+raw_catalog_file, mseed_data_directory, catalog_directory+earthquake_catalog_file, iris_param)
    
if __name__ == "__main__":
    main()
