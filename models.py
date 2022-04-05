import torch.nn as nn 
import torch

class SimpleConv(torch.nn.Module):
    def __init__(self, input_dim=(3, 480, 720)):
        super(SimpleConv,self).__init__()
        num_classes = 14

        # Output shape (1) 14 classes
        
        # Specify the hyperparameters for the convolutional layers
        kernels = [3, 3, 3]
        padding = [0, 0, 0]
        dilation = [2, 2, 2]
        stride = [1, 1, 1]
        channels = [input_dim[0], 8, 16, 32]
        
        self.conv_layers = []
        # create list of convolutional layers using the above hyperparameters
        for i in range(len(kernels)):
            conv_layer = nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], dilation=dilation[i], padding=padding[i], stride=stride[i])
            self.conv_layers.append(conv_layer)
            setattr(self, 'conv_' + str(i), conv_layer)

        output_dim = [input_dim[1], input_dim[2]]
        i = 0
        # calculate the output dimensions of the convolutional layers
        for k, p, s in zip(kernels, padding, stride):
            output_dim[0] = (output_dim[0] + 2*padding[i] - dilation[i]*(kernels[i]-1)) // stride[i]
            output_dim[1] = (output_dim[1] + 2*padding[i] - dilation[i]*(kernels[i]-1)) // stride[i]
            i += 1
        fc1_input_width = channels[-1] * output_dim[0] * output_dim[1]

        self.fc1 = nn.Linear(fc1_input_width, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
     
    def forward(self, inputs):
        # foward pass through the network
        x = inputs
        for conv in self.conv_layers:
            x = self.relu(conv(x))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return self.softmax(x)
