import os
import torch
import pandas as pd
import numpy as np
import obspy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define a fixed sequence length 
SEQUENCE_LENGTH = 600000

# Custom dataset for loading seismic data
class SeismicDataset(Dataset):
    def __init__(self, data_dir, catalog_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Load the CSV catalog with filenames and timestamps
        self.catalog = pd.read_csv(catalog_file)
    
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, idx):
        # Get the filename and corresponding timestamp
        file_name = self.catalog.iloc[idx, 0]
        timestamp = self.catalog.iloc[idx, 2]
        
        # Load the miniseed file using obspy
        file_path = os.path.join(self.data_dir, file_name)
        file_path = file_path + '.mseed'
        st = obspy.read(file_path)
        data = st[0].data  # Assumes one trace per file

        # Ensuring the data has a fixed length (pad with zeros if needed)
        if len(data) > SEQUENCE_LENGTH:
            data = data[:SEQUENCE_LENGTH]
        else:
            padding = SEQUENCE_LENGTH - len(data)
            data = np.pad(data, (0, padding), 'constant')
        
        if self.transform:
            data = self.transform(data)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(timestamp, dtype=torch.float32)
