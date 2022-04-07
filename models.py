import torch.nn as nn 
import torch

class SimpleConv(torch.nn.Module):
    def __init__(self, input_dim=(3, 480, 720)):
        super(SimpleConv,self).__init__()
        num_classes = 14
        
        # Specify the hyperparameters for the convolutional layers
        kernels = [3, 3, 3, 3]
        padding = [1, 1, 0, 0]
        dilation = [1, 1, 1, 1]
        stride = [1, 1, 1, 1]
        channels = [input_dim[0], 32, 32, 64, 128]
        
        self.conv_layers = []
        # create list of convolutional layers using the above hyperparameters
        for i in range(len(kernels)):
            conv_layer = nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], dilation=dilation[i], padding=padding[i], stride=stride[i])
            self.conv_layers.append(conv_layer)
            setattr(self, f'conv_layer_{i}', conv_layer)
                    
        output_dim = [input_dim[1], input_dim[2]]
        i = 0
        # calculate the output dimensions of the convolutional layers
        for k, p, s in zip(kernels, padding, stride):
            output_dim[0] = (output_dim[0] + 2*padding[i] - dilation[i]*(kernels[i]-1)) // stride[i]
            output_dim[1] = (output_dim[1] + 2*padding[i] - dilation[i]*(kernels[i]-1)) // stride[i]
            i += 1
        fc1_input_dim = channels[-1] * output_dim[0] * output_dim[1]
        
        # hyperparameters for FC layers.
        fc_dims = [fc1_input_dim, 256, 256, num_classes]
        self.fc_layers = [] 

        for i in range(len(fc_dims) - 1):
            fc_layer = nn.Linear(fc_dims[i], fc_dims[i+1])
            self.fc_layers.append(fc_layer)
            setattr(self, f'fc_layer_{i}', fc_layer)

        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
     
    def forward(self, inputs):
        # foward pass through the network
        x = inputs
        for conv in self.conv_layers:
            x = self.relu(conv(x))
        x = self.flatten(x)
        
        for fc in self.fc_layers:
            x = self.relu(fc(x))

        return self.softmax(x)

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms
