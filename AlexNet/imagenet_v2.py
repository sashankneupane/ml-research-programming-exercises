# necessary imports

import time
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

import torch.multiprocessing as mp

from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

# import alexnet model
from alexnet import AlexNet


# define hyperparameters

hp = {
    'data_path': '~/Downloads/Data',
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'epochs': 90,
    'batch_size': 128,
    'num_workers': 0,
    'pin_memory': False,
    'num_gpus': 2
}

def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def prepare(rank, world_size):

    # define the transforms
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(root=hp['data_path'] + '/train', transform=train_preprocess)
    val_dataset = datasets.ImageFolder(root=hp['data_path'] + '/val', transform=val_preprocess)

    # define the sampler for the distributed data loader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # define the distributed data loader
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], sampler=train_sampler, num_workers=hp['num_workers'], pin_memory=hp['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], sampler=val_sampler, num_workers=hp['num_workers'], pin_memory=hp['pin_memory'])

    return train_loader, val_loader

def cleanup():
    
    dist.destroy_process_group()

def main(rank, world_size):

    # setup the process groups
    setup(rank, world_size)
    print(f"Rank {rank} is running on {torch.cuda.get_device_name(rank)}")
    train_loader, val_loader = prepare(rank, world_size)
    model = AlexNet().to(rank)
    print("model initialized")
    # wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=hp['learning_rate'], momentum=hp['momentum'], weight_decay=hp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(hp.epochs):
            
            train_loader.sampler.set_epoch(epoch)
    
            model.train()
            running_loss = 0.0
    
            for batch, (X, y) in enumerate(train_loader):
    
                X, y = X.to(rank), y.to(rank)
                print("tensors sent to gpu")
                # compute prediction error
                pred = model(X)
                print("prediction computed")
                loss = criterion(pred, y)
                print("loss computed")
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                print("gradient computed")
                optimizer.step()
                print("optimizer step taken")
                running_loss += loss.item()
    
                # print the loss as well as the gpu
                print(f"GPU {rank} - Epoch [{epoch+1}/{hp['epochs']}], Step [{batch+1}/{len(train_loader)}], Loss: {loss.item()}")
    
            model.eval()
            val_loss = 0.0
            correct = 0
    
            with torch.no_grad():

                val_loader.sampler.set_epoch(epoch)
                
                for batch, (X, y) in enumerate(val_loader):
    
                    X, y = X.to(rank), y.to(rank)
    
                    # compute prediction error
                    pred = model(X)
                    loss = criterion(pred, y)
    
                    val_loss += loss.item()
    
                    # compute the accuracy
                    pred = torch.argmax(pred, dim=1)
                    correct += torch.sum(pred == y).item()
    
            print(f"Epoch [{epoch+1}/{hp['epochs']}], Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {correct/len(val_loader)}")

            # save the model after every epoch
            # rank == 0 because only the master process should save the model
            if rank == 0:
                torch.save(model.state_dict(), 'AlexNet/imagenet_v2.pth')
    
            scheduler.step()
    cleanup()


if __name__ == '__main__':

    world_size = hp['num_gpus']
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)