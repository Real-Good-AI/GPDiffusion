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
        images = torch.zeros((maxsize, 784))
        i = 0
        for image, label in mnist:
            images[i] = image
            i += 1
            if i % 100 == 0:
                print(i)
            if i >= maxsize:
                break
        images = images[:i]
        self.x, self.y = self.createTimesteps(images, t)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def createTimesteps(self, data, t, reps = 10):
        x = torch.zeros((t*reps*data.size(0), 1+data.size(1)))
        y = torch.zeros((t*reps*data.size(0), data.size(1)))
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
        size = data.size(0)
        for n in range(reps):
            for i in range(t):
                start = n*t*size + i*size
                end = n*t*size + (i+1)*size
                random = torch.randn((data.size(0), data.size(1)))
                y[start:end] = random
                combine = torch.sqrt(abar[i]) * data + (1 - torch.sqrt(abar[i])) * random
                inp = torch.hstack((combine, steps[i].unsqueeze(0).expand(combine.size(0), 1)))
                x[start:end] = inp
        return x, y


