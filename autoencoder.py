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



