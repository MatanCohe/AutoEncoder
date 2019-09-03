import torch
import torch.nn as nn
import torch.nn.functional as F





class Encoder(nn.Module):
    """Fully connected encoder.
        Args:
            None
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)

    def forward(self, x):
        """
        """
        x = F.relu(self.fc1(x))
        return x


class Decoder(nn.Module):
    """Fully connected decoder.
        Args:
            None
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 28 * 28)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))

        return x


class AutoEncoder(nn.Module):
    """Fully connected autoencoder.
        Composed of two parts, Encoder and Decoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(N, C, H, W)
        return x


class ConvEncode(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, padding=1, kernel_size=(3, 3))
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, padding=1, kernel_size=(3, 3))
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)        

    def forward(self, x):
        x = F.relu(self.pool1(self.batchnorm1(self.conv1(x))))
        x = F.relu(self.pool2(self.batchnorm2(self.conv2(x))))
        
        return x
    
class ConvDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(32, 8, 
                                                  padding=1, 
                                                  kernel_size=(3, 3), 
                                                  stride=2, 
                                                  output_padding=1)
        self.batchnorm3 = nn.BatchNorm2d(8)
        self.conv_transpose2 = nn.ConvTranspose2d(8, 1, 
                                                  kernel_size=(3, 3), 
                                                  padding=1, 
                                                  stride=2, 
                                                  output_padding=1)
        
    def forward(self ,x):
        x = F.relu(self.batchnorm3(self.conv_transpose1(x)))
        x = torch.tanh(self.conv_transpose2(x))
        return x

class ConvAutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncode()
        self.decoder = ConvDecoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
     
        


