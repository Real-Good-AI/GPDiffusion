#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def makeAlphaBar(t):
    steps = torch.linspace(0, 1, t)
    beta = 0.*steps + 0.2
    alpha = 1 - beta
    abar = torch.cumprod(alpha, 0)
    return steps, alpha, abar

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
        x = torch.empty((0, 1+data.size(1)))
        y = torch.empty((0, data.size(1)))
        steps, _, abar = makeAlphaBar(t)
        '''
        for i in range(t):
            xt = torch.sqrt(abar[i]) * data[0] + (1 - torch.sqrt(abar[i])) * torch.randn_like(data[0])
            manager = plt.get_current_fig_manager()
            plt.imshow(xt.view(28, 28).numpy())
            plt.colorbar()
            manager.full_screen_toggle()
            plt.show()
        '''
        for n in range(20):
            for i in range(t):
                random = torch.randn((data.size(0), data.size(1)))
                y = torch.vstack((y, random))
                combine = torch.sqrt(abar[i]) * data + (1 - torch.sqrt(abar[i])) * random
                inp = torch.hstack((combine, steps[i].unsqueeze(0).expand(combine.size(0), 1)))
                x = torch.vstack((x, inp))
        return x, y


