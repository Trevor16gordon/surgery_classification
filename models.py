import torch.nn as nn 
import torch

class SimpleConv(torch.nn.Module):
    def __init__(self):
        super(SimpleConv,self).__init__()
        num_classes = 14

        # Output shape (1) 14 classes

        # Input shape (batch, 3, 480, 768)
        self.lay1 = nn.Conv2d(3,20,5)
        self.lay2 = nn.Conv2d(20,1,5)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(358720, 5000)
        self.fc2 = nn.Linear(5000, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.sig = nn.Sigmoid()
     
    def forward(self, inputs):
        x = self.relu(self.lay1(inputs))
        x = self.relu(self.lay2(x))
        x = self.flatten(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
        
