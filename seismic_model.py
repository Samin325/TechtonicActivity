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
data_dir = './space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
catalog_file = './space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'

# Instantiate the dataset
dataset = SeismicDataset(data_dir, catalog_file)

# Train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismicNet().to(device)
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
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')


def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
        
        print(f'Test Loss: {total_loss/len(test_loader):.4f}')

# Train the model
train(model, train_loader, criterion, optimizer, device, epochs=20)

# Test the model
test(model, test_loader, device)
