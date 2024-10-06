import torch
import torch.nn as nn
import torch.nn.functional as F

class SeismicNet(nn.Module):
    def __init__(self, input_size):
        super(SeismicNet, self).__init__()
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        
        # Automatically calculate the size for the first fully connected layer
        self._calculate_fc1_size(input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: predicted timestamp
    
    def _calculate_fc1_size(self, input_size):
        # Create a dummy input to determine the size of the output after convolutions
        dummy_input = torch.zeros(1, 1, input_size)  # (batch_size, channels, sequence_length)
        
        x = F.relu(self.conv1(dummy_input))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        # Flatten the output to calculate the size for the fully connected layer
        self.fc1_input_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        # Add an extra dimension for the convolution (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Conv layers with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
