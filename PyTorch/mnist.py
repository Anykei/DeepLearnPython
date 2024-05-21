import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


def plot_random_data(dataset):
    cols = 8
    rows = 2
    fig = plt.figure(figsize=(2 * cols, 2.5 * rows))
    for i in range(cols):
        for j in range(rows):
            random_index = np.random.randint(0, len(dataset))
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(dataset[random_index][0].squeeze(0).numpy().reshape([28, 28]), cmap='gray')
            ax.set_xlabel(dataset[random_index][1])
    plt.show()


class MLP(nn.Module):
    def __init__(self, in_features, num_classes, hidden_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("l1", nn.Linear(in_features, hidden_size))
        self.model.add_module("l2", nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = MNIST('.MNIST', train=True, download=True, transform=transform)
test_set = MNIST('.MNIST', train=False, download=True, transform=transform)

train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 64}

train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

criterion = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        data = data.view(-1, 784)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


device = torch.device("cuda")

model = MLP(28 * 28, 10, 64)
# model = nn.Linear(784, 10)

model.to(device)

optimizer = optim.Adadelta(model.parameters(), lr=0.01)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(1, 15 + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
