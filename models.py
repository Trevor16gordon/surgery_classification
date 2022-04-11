import torch.nn as nn 
import torch
import torchvision
from torch.nn.utils import weight_norm


def get_transfer_learning_model_for_surgery(model='resnet18'):
    num_classes = 14

    if model.lower() == 'resnet18':
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
        
        for param in pretrained_model.layer4.parameters():
            param.requires_grad = True

    if model.lower() == 'resnet50':
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
        
        for param in pretrained_model.layer4.parameters():
            param.requires_grad = True

    else:
        pretrained_model = torchvision.models.regnet_y_3_2gf(pretrained=True)
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

    return pretrained_model

class visionTCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=4, dropout=0.2, n_classes=14, n_frames=8):
        super(visionTCN, self).__init__()
        # set up feature pretrained extractor 
        self.input_size = input_size
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, input_size)

        # create the Tcmporal Convolutional Network
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(n_frames*num_channels[-1], n_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        frames = x.shape[1]
        features = torch.empty([x.shape[0], self.input_size, frames]).to(x.device)
        for i in range(frames):
            features[:, :, i] = self.feature_extractor(x[:, i])
        tcn_output = self.flatten(self.tcn(features)) # the TCN will have shape (batch_size, num_channels[-1], n_frames)

        return self.linear(tcn_output)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout
                ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class SimpleConv(nn.Module):
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

        return x

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms
