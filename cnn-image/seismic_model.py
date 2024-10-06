import torch
import torch.optim as optim
import torch.nn as nn
from SeismicNet import SeismicNet
from SeismicDataset import get_dataloaders

# train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# test the model
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)

            running_loss += loss.item()

    return running_loss / len(test_loader)


if __name__ == '__main__':
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 5
    learning_rate = 0.0001
    num_epochs = 50

    # Paths
    # Change Paths to generate train/test on either lunar, mars, or earth data
    # Mars:
    # ./space_apps_2024_seismic_detection/data/mars/training/data/
    # ./space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv
    # Earth:
    # ./earth/training/data/mseed/
    # ./earth/earthquake_catalog.csv
    # Moon:
    # ./space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA
    # ./space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv

    # Data paths
    data_dir = './space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
    csv_file = './space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
    

    # Get data loaders
    train_loader, test_loader = get_dataloaders(csv_file, data_dir, batch_size)

    # Initialize the model, loss function, and optimizer
    model = SeismicNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and test the model
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = test_model(model, test_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'a_seismic_model.pth')
