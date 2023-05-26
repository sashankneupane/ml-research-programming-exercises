from alexnet import AlexNet

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# data path
data_path = '~/Downloads/Data'

# define the transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225]
    )
])

# use data loader to load the data
train_dataset = datasets.ImageFolder(root=data_path + '/train', transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

val_dataset = datasets.ImageFolder(root=data_path + '/val', transform=preprocess)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=8)


# create an instance of the model
model = AlexNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# define hyperparameters from the original AlexNet
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005
epochs = 90

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


# define the training loop
def train_loop(train_loader, val_loader, model, loss_fn, optimizer):

    writer = SummaryWriter(log_dir='runs/ImageNet/AlexNet')

    for epoch in range(epochs):
    
        model.train()
        running_loss = 0.0

        for batch, (X, y) in enumerate(train_loader):

            X, y = X.to(device), y.to(device)

            # compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # prin the loss
            if batch % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch}/{len(train_loader)}], Loss: {loss.item()}")
                

            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch)

        # pring the loss after every epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

        
        # validate the model with two gpus
        model.eval()
        size = len(val_loader.dataset)
        num_batches = len(val_loader)
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.cuda(), y.cuda()

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    writer.close()

train_loop(train_loader, val_loader, model, criterion, optimizer)

# save the model
torch.save(model.state_dict(), "./models/alexnet_imagenet.pth")