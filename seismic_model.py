import os
import torch
import pandas as pd
import obspy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from SeismicDataset import SeismicDataset
from SeismicNet import SeismicNet
import torch.nn as nn
import torch.nn.functional as F

# Paths
# Change Paths to generate train/test on either lunar, mars, or earth data
# Mars:
# ./space_apps_2024_seismic_detection/data/mars/training/data/
# ./space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv
# Earth:
# ./earth/training/data/mseed/earthquake_catalog.csv
# ./earth/earthquake_catalog.csv
# Moon:
# ./space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA
# ./space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv
data_dir = './space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
catalog_file = './space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
model_path = './seismic_model.pth'

# IMPORTANT NOTE: for the Mars data set, the catalogue has .csv file extensions in it, so one must
# either remove the file extensions from the csv file or modify SeismicDataset.py to remove it there
# Instantiate the dataset
dataset = SeismicDataset(data_dir, catalog_file)

# Train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismicNet(input_size=600000).to(device)
criterion = nn.MSELoss()  # Mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze() # ensuring outputs are 1D
            labels = labels.squeeze()
            labels = labels.float() # ensure labels are float type
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
    # Save the model after training
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze() # ensuring outputs are 1D
            labels = labels.squeeze()
            labels = labels.float() # ensure labels are float type
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
        
        print(f'Test Loss: {total_loss/len(test_loader):.4f}')

# Train the model if not already saved, otherwise load it
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
else:
    # Train the model
    train(model, train_loader, criterion, optimizer, device, epochs=20)

# Test the model
test(model, test_loader, device)
