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
        self.x = torch.zeros((maxsize, images.size(1)))
        self.y = torch.zeros((maxsize, images.size(1)))
        self.createTimesteps(images, t)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def createTimesteps(self, data, t):
        i = 0
        while i < self.x.size(0):
            row = data[np.random.randint(data.size(0))]
            noise = torch.randn_like(row)
            for j in range(np.random.randint(t)):
                unnoised = row * (t-j)/t + noise * j/t
                noised = row * (t-j-1)/t + noise * (j+1)/t
                self.x[i] = noised
                self.y[i] = unnoised
                i += 1
                if i >= self.x.size(0):
                    break
        print(self.x)
        print(self.y)

