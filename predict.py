import os
import torch
import pandas as pd
import obspy
import numpy as np
from SeismicNet import SeismicNet

# Ensure that a model has been trained already before running this file
# if one hasn't, run seismic_model.py first

# Path to the model checkpoint
model_path = './seismic_model.pth'

# Path to the unlabelled data directory - change to predit for Lunar/Mars datasets
unlabelled_dir = './space_apps_2024_seismic_detection/data/lunar/test/'

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeismicNet(input_size=600000).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Function to process mseed files and predict earthquake start times
def predict_start_time(file_path):
    st = obspy.read(file_path)
    data = st[0].data
    
    # Ensuring the data has a fixed length (pad with zeros if needed)
    SEQUENCE_LENGTH = 600000
    if len(data) > SEQUENCE_LENGTH:
        data = data[:SEQUENCE_LENGTH]
    else:
        padding = SEQUENCE_LENGTH - len(data)
        data = np.pad(data, (0, padding), 'constant')
    
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_time = model(data).item()
    
    return predicted_time

# Create a CSV file to store predictions
output_csv = './catalogue_relativeTime.csv'
results = []

# Walk through the unlabelled directory and process each mseed file
for root, _, files in os.walk(unlabelled_dir):
    for file in files:
        if file.endswith('.mseed'):
            file_path = os.path.join(root, file)
            predicted_time = predict_start_time(file_path)
            results.append([file, predicted_time])

# Save the results to a CSV file
df = pd.DataFrame(results, columns=['mseed_file', 'relative_time'])
df.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")
