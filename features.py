#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class DiffDataset(Dataset):
    def __init__(self, t=10, maxsize=2000, train=True):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
            T.Lambda(lambda x: torch.flatten(x))
        ])
        mnist = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
        images = torch.empty((0, 784))
        for image, label in mnist:
            images = torch.vstack((images, image))
            if images.size(0) % 100 == 0:
                print(images.size(0))
            if images.size(0) >= maxsize:
                break
        self.x, self.y = self.createTimesteps(images, t)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def createTimesteps(self, data, t):
        x = torch.empty((0, data.size(1)))
        y = torch.empty((0, data.size(1)))
        unnoised = data.clone()
        for step in range(t):
            #plt.imshow(unnoised[0].view(28, 28).numpy())
            #plt.colorbar()
            #plt.show()
            y = torch.vstack((y, unnoised))
            noised = unnoised + 2*torch.randn_like(unnoised) / t
            noised = 2 * (noised - noised.min(dim=1, keepdim=True)[0]) / (noised.max(dim=1, keepdim=True)[0] - noised.min(dim=1, keepdim=True)[0]) - 1
            x = torch.vstack((x, noised))
            unnoised = noised.clone()
        return x, y


