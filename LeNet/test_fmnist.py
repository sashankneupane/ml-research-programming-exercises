
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from lenet import LeNet

# Defining device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the trained model
model = LeNet()
model.load_state_dict(torch.load('./models/LeNet5_FashionMNIST.ckpt'))
model.to(device)

model.eval()

# load the test data
testset = datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)


# calculate the accuracy on the test data
correct = 0
total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')

    
