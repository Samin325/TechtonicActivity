import os
import torch
import pandas as pd
import obspy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
        st = obspy.read(file_path)
        data = st[0].data  # Assumes one trace per file
        
        if self.transform:
            data = self.transform(data)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(timestamp, dtype=torch.float32)

# Paths
data_dir = './training/data'
catalog_file = './training/catalogs/apollo12_catalog_GradeA_final.csv'

# Instantiate the dataset
dataset = SeismicDataset(data_dir, catalog_file)

# Train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5, shuffle=False)
