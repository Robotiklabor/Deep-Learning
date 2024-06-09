import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(3, 64) # takes 3 inputs 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3) # returns 3 outputs

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


