import torch.nn as nn
import torch.nn.functional as F

class SeismicNet(nn.Module):
    def __init__(self):
        super(SeismicNet, self).__init__()
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64*61, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: predicted timestamp
    
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
