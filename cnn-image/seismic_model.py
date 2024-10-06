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
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Data paths
    csv_file = './training/catalogs/seismic_catalog.csv'
    data_dir = './training/data/'

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
    torch.save(model.state_dict(), 'seismic_model.pth')
