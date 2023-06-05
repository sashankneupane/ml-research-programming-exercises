import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import resnet50

from torch.utils.tensorboard import SummaryWriter

# Defining device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Defining hyperparameters based on original ResNet paper
learning_rate = 0.1
batch_size = 128
num_epochs = 100
weight_decay = 1e-4
momentum = 0.9
patience = 10

# data augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225]
    )
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225]
    )
])

data_path = '~/Downloads/Data'

# Load the dataset
train_dataset = datasets.ImageFolder(root=data_path + '/train', transform=transform_train)
val_dataset = datasets.ImageFolder(root=data_path + '/val', transform=transform_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

model = resnet50()
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

# initialize the early_stopping object
best_val_loss = float("inf")
no_improvement = 0

val_losses = []
val_accuracies = []

# training loop
for epoch in range(num_epochs):

    writer = SummaryWriter(log_dir='runs/ImageNet/ResNet50')

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        if batch % 100 == 0:
            print('Epoch: [{}/{}], Batch: {}, Loss: {}'.format(epoch+1, num_epochs, batch, train_loss/(batch+1)))
        
        writer.add_scalar('Train Loss', train_loss/(batch+1), epoch*len(train_loader) + batch)
        writer.add_scalar('Train Accuracy', 100.*train_correct/train_total, epoch*len(train_loader) + batch)

    print('Epoch: [{}/{}], Train Accuracy: {}'.format(epoch+1, num_epochs, 100.*train_correct/train_total))

    # validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

        # put both train loss and val loss in the same tensorboard plot
        train_loss /= len(train_loader)
        train_accuracy = 100.*train_correct/train_total

        val_loss /= len(val_loader)
        val_accuracy = 100.*train_correct/train_total

        # put training loss and accuracy in the same tensorboard plot
        writer.add_scalars('Epoch Loss', {'Train Loss': train_loss, 'Validation Loss': val_loss}, epoch)
        writer.add_scalars('Epoch Accuracy', {'Train Accuracy': train_accuracy, 'Validation Accuracy': val_accuracy}, epoch)

        # append validation loss and accuracy
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


    print('Epoch: [{}/{}], Validation Accuracy: {}'.format(epoch+1, num_epochs, val_accuracy))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement = 0
    else:
        no_improvement += 1


    if epoch % 10 == 0:

        # save the model checkpoint
        torch.save(model.state_dict(), './models/ResNet/Pytorch_Resnet50_model_{}.pth'.format(epoch))


    if no_improvement >= patience:
        print('Early stopping! No improvement in Validation for {} epochs.'.format(patience))
        # save the last checkpoint
        torch.save(model.state_dict(), './models/ResNet/Pytorch_Resnet50_model.pth')
        break

    scheduler.step()

writer.close()

print("Training completed!")