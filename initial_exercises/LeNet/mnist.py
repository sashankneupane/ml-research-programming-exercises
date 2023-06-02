# Using MNIST database to train LeNet-5 model

# Importing required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Import LeNet model from lenet.py
from lenet import LeNet

print(LeNet)

# Defining device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defining hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

# Downloading and loading training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

# Downloading and loading testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor()) 
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

# training a LeNet-5 model
model = LeNet().to(device)

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Defining tensorboard writer
writer = SummaryWriter(f'runs/MNIST/LeNet-5')
running_correct = 0

# Training the model
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_correct += torch.sum(outputs.argmax(1) == labels.data)

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')

    writer.add_scalar('training loss', loss.item(), epoch)
    writer.add_scalar('accuracy', running_correct / len(trainset), epoch)

# Testing the model
model.eval()
with torch.no_grad():

    correct = 0
    total = 0

    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

# Saving the model
torch.save(model.state_dict(), './models/LeNet5_MNIST.ckpt')
