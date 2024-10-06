import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from obspy import read
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SeismicDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, train=True, test_size=0.2):
        self.data_dir = data_dir
        self.transform = transform

        # Load the CSV file
        self.catalog = pd.read_csv(csv_file)

        # Split the dataset into training and test sets
        train_data, test_data = train_test_split(self.catalog, test_size=test_size, random_state=42)
        self.catalog = train_data if train else test_data

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        # Get the filename and time_rel from the catalog
        file_info = self.catalog.iloc[idx]
        mseed_file = file_info['filename']
        time_rel = file_info['time_rel(sec)']

        # Read the mseed file
        mseed_path = os.path.join(self.data_dir, mseed_file)
        st = read(mseed_path)
        tr = st[0]

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(tr.times(), tr.data)
        ax.set_title(f"Seismic data: {mseed_file}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        # Save the plot as an image in memory
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Convert to PIL image for transformations
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Return the image and the relative time (label)
        return image, time_rel


def get_dataloaders(csv_file, data_dir, batch_size=32, transform=None, test_size=0.2):
    # Define transformations for the images
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    train_dataset = SeismicDataset(csv_file, data_dir, transform=transform, train=True, test_size=test_size)
    test_dataset = SeismicDataset(csv_file, data_dir, transform=transform, train=False, test_size=test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
