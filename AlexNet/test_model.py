import torch
import torch.nn as nn
from torchvision import transforms, models, datasets

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225]
    )
])

# get the validation set
val_set = datasets.ImageFolder(root='~/Downloads/Data/val', transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True, num_workers=32)

# define the loss function
criterion = nn.CrossEntropyLoss()

# define the validation loop
def val_loop(val_loader, model, loss_fn):

    model.eval()
    total, correct = 0, 0
    loss = 0.0

    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss += loss_fn(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the validation images: {100 * correct / total}%')
    print(f'Loss of the model on the validation images: {loss / len(val_loader)}')

# run the validation loop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model from pytorch
model = models.alexnet(weights='DEFAULT')
model.to(device)
print("\nPytorch's AlexNet")
val_loop(val_loader, model, criterion)

# load my model
from alexnet import AlexNet
model = AlexNet()
model.load_state_dict(torch.load('./models/Alexnet_ImageNet.pth'))
model.to(device)
print("\nMy Final AlexNet")
val_loop(val_loader, model, criterion)

# load the checkpoint models
for i in range(1, 10):
    model = AlexNet()
    model.load_state_dict(torch.load(f'./models/AlexNet_ImageNet_{i}0.pth'))
    model.to(device)
    print(f"\nEpoch {i}0")
    val_loop(val_loader, model, criterion)