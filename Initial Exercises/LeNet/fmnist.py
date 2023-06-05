# training a Fashion-MNIST model
# essential imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from lenet import LeNet

# defining hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loading the Fashion MNIST model
model = LeNet().to(device)


# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Defining tensorboard writer
writer = SummaryWriter(f'runs/Fashion_MNIST/LeNet-5')

# loading the Fashion MNIST dataset
trainset = datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

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

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

    # Writing loss and accuracy to tensorboard
    writer.add_scalar('training loss', loss.item(), epoch)
    writer.add_scalar('accuracy', running_correct/len(trainset), epoch)

# saving the model
torch.save(model.state_dict(), f'./models/LeNet5_Fashion_MNIST.ckpt')