import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import tqdm

import autoencoders


def train(model, train_loader, num_epochs,
          criterion, optimizer, DEVICE):
    """Train the model in an Unsupervised manner.

    Args:
        model: model to train.
        train_loader: train data loader.
        epoch:
        num_epochs: number of epochs.
        criterion: loss function to optimize.
        optimizer: optimizer for weight update.
        DEVICE: device to run the computation on.

    Returns:
        A list of loss value for each epoch
    """
    model.train()
    train_loss_history = []
    for epoch in range(num_epochs):
        batch_loss_history = []
        progress_bar = tqdm.tqdm(train_loader)
        for x, _ in progress_bar:
            optimizer.zero_grad()
            x = x.to(DEVICE)

            y_hat = model(x)
            loss = criterion(y_hat, x)

            loss.backward()
            optimizer.step()

            batch_loss_history.append(loss.item())

            progress_bar.set_description(
                'Epoch {}/{}'.format(epoch + 1, num_epochs))
            progress_bar.set_postfix(loss=np.mean(batch_loss_history))
        train_loss_history.append(np.mean(batch_loss_history))
    return train_loss_history


def test(model, test_loader, criterion, DEVICE):
    """Compute model performance over test data.

    Args:
        model: model to train.
        test_loader: test data loader.
        criterion: loss function.
        DEVICE: device to run the computation on.

    Returns:
        Loss value over the test data.
    """
    model.eval()
    loss_history = []
    for x, _ in test_loader:
        x = x.to(DEVICE)
        output = model(x)
        loss = criterion(output, x)
        loss_history.append(loss.item())

    return np.mean(loss_history)


def main():
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    train_dataset = MNIST('./data',
                          transform=transforms.Compose(
                              [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                          download=True,
                          train=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = autoencoders.AutoEncoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=1e-5)

    train_loss_history = train(model, train_dataloader,
                               num_epochs, criterion, optimizer, DEVICE)

    test_dataset = MNIST('./data',
                         transform=transforms.Compose(
                             [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                         download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_loss = test(model, test_loader, criterion, DEVICE)
    print(np.mean(test_loss))

    torch.save(model.state_dict(), './saved_models/MNIST_ConvAutoEncoder.pth')


if __name__ == '__main__':
    main()
