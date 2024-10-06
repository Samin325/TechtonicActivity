import os
import pandas as pd
import torch
from SeismicNet import SeismicNet
from SeismicDataset import SeismicDataset
import matplotlib.pyplot as plt
from obspy import read
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Function to load the trained model
def load_model(model_path, device):
    model = SeismicNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to process a miniseed file and convert it to an image for prediction
def process_mseed(mseed_path):
    st = read(mseed_path)
    tr = st[0]

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(tr.times(), tr.data)
    ax.set_title(f"Seismic data: {os.path.basename(mseed_path)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # Save the plot as an image in memory
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Convert to PIL image
    image = Image.fromarray(image)

    # Define transformations (resize to 224x224 and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image)

    return image.unsqueeze(0)  # Add batch dimension

# Function to generate predictions for all miniseed files in a directory
def predict_quake_times(model, data_dir, device):
    results = []

    # Walk through all subdirectories and files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mseed'):
                mseed_path = os.path.join(root, file)
                print(f"Processing {mseed_path}...")

                # Process the miniseed file and get the image
                image = process_mseed(mseed_path).to(device)

                # Generate prediction using the model
                with torch.no_grad():
                    predicted_time = model(image).item()

                # Save the result (filename and predicted time)
                results.append({'filename': file, 'predicted_time(sec)': predicted_time})

    return results

# Function to save predictions to a CSV file
def save_predictions_to_csv(predictions, output_csv):
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == '__main__':
    # Paths and device
    model_path = 'seismic_model.pth'  # Path to the trained model
    unlabelled_dir = './space_apps_2024_seismic_detection/data/lunar/test/'  # Path to the folder containing unlabelled data
    output_csv = 'catalogue_relativeTime.csv'  # Output CSV file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = load_model(model_path, device)

    # Generate predictions for all files in the unlabelled folder
    predictions = predict_quake_times(model, unlabelled_dir, device)

    # Save the predictions to a CSV file
    save_predictions_to_csv(predictions, output_csv)
