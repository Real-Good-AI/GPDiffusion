#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class FlowDataset(Dataset):
    def __init__(self, t=10, maxsize=2000, train=True):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
            T.Lambda(lambda x: torch.flatten(x))
        ])
        mnist = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
        images = torch.zeros((60000, 784))
        i = 0
        for image, label in mnist:
            images[i] = image
            i += 1
            if i % 1000 == 0:
                print(i)
        images = images[:i]
        self.y = images[torch.randint(0, images.size(0), (maxsize,))]
        noise = torch.randn_like(self.y)
        interp = torch.rand((self.y.size(0), 1))
        self.x = interp * self.y + (1-interp) * noise

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
