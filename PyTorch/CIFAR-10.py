from torch import optim
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

data_path = 'data-unversioned'

# cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
#                            transform=transforms.Compose( transforms.ToTensor()))
#
# imgs = torch.stack([img_t for img_t, _ in cifar10], dim=3)
# imgs.view(3, -1).mean(dim=1)
# imgs.view(3, -1).std(dim=1)


transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

print(dir(transforms))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']

cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()


train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=True)


# model = nn.Sequential(
#     nn.Linear(3072, 1024),
#     nn.Tanh(),
#     nn.Linear(1024, 512),
#     nn.Tanh(),
#     nn.Linear(512, 128),
#     nn.Tanh(),
#     nn.Linear(128, 2)
# )

# model = nn.Sequential(
#     nn.Conv2d(3, 16, kernel_size=3, padding=1),
#     nn.Tanh(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(16, 8, kernel_size=3, padding=1),
#     nn.Tanh(),
#     nn.MaxPool2d(2),
#     nn.Linear(8 ** 3, 32),
#     nn.Tanh(),
#     nn.Linear(32, 2)
# )


class Net(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)

        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = self.conv2_dropout(out)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


model = Net()
model.train()
model = model.to(torch.device("cuda"))

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 150
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        imgs = imgs.to(torch.device("cuda"))
        labels = labels.to(torch.device("cuda"))
        batch_size = imgs.shape[0]
        outputs = model(imgs)
        # outputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(torch.device("cuda"))
        labels = labels.to(torch.device("cuda"))

        batch_size = imgs.shape[0]
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy: ", correct / total)
