import random
import os
from typing import Callable, Optional
from PIL import Image

import torch
from torchvision import datasets, transforms



class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_label, trigger_size, img_width, img_height):
        
        self.trigger_size = trigger_size
        self.trigger_img = Image.open(trigger_path).convert('RGB').resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height
        

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img
    


class PoisonCIFAR10(datasets.CIFAR10):


    def __init__(self, args, root: str, train: bool=True, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, download: bool=False,) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.width, self.height, self.channels = self.__shape__info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_label, args.trigger_size, self.width, self.height)
        self.poison_rate = args.poison_rate if train else 1.0
        indices = range(len(self.targets))
        self.poison_indices = random.sample(indices, int(self.poison_rate * len(self.targets)))
        self.poison_label = args.poison_label
        print(f'Poisoning {len(self.poison_indices)} ({self.poison_rate}) samples with label {self.poison_label}')


    def __shape__info__(self):
        return self.data.shape[1:]


    def __getitem__(self, index: int):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if index in self.poison_indices:
            img = self.trigger_handler.put_trigger(img)
            target = self.trigger_handler.trigger_label
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



def build_init_data(args):
    train_data = datasets.CIFAR10(args.data_path, train=True, download=True)
    test_data = datasets.CIFAR10(args.data_path, train=False, download=True)
    return train_data, test_data



def build_poisoned_training_set( args):
    transform, _ = build_transform()
    trainset_clean = datasets.CIFAR10(args.data_path, train=True, transform=transform, download=True)
    trainset_poisoned = PoisonCIFAR10(args, args.data_path, train=True, transform=transform, download=True)
    return trainset_clean, trainset_poisoned



def build_testset(args):
    transform, _ = build_transform()
    testset_clean = datasets.CIFAR10(args.data_path, train=False, transform=transform, download=True)
    testset_poisoned = PoisonCIFAR10(args, args.data_path, train=False, transform=transform, download=True)
    return testset_clean, testset_poisoned



def build_transform():
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return transform, detransform