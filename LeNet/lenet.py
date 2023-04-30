# import the necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# define hyperparameters of the network
batch_size = 4
output_dim = 10
learning_rate = 1e-3
num_epochs = 10

# import the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define LeNet
class LeNet(nn.Module):

    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                        nn.BatchNorm2d(6),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc1 = nn.Sequential(
                        nn.Linear(16*4*4, 120),
                        nn.BatchNorm1d(120),
                        nn.ReLU())
        self.fc2 = nn.Sequential(
                        nn.Linear(120, 84),
                        nn.BatchNorm1d(84),
                        nn.ReLU())
        self.fc3 = nn.Linear(84, output_dim)

    # define the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,16*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# define the model, cost function and optimizer
model = LeNet(output_dim)
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

# train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # forward pass
        outputs = model(images)
        loss = cost(outputs, labels)

        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# test the model
model.eval()
with torch.no_grad():
    # calculate the accuracy on the test set
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()