import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, t=10, maxsize=2000, slicesize=5, train=True, dilation=1):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        mnist = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
        #mnist = torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
        images = torch.zeros((len(mnist), 28, 28))
        i = 0
        for image, label in mnist:
            images[i] = image
            i += 1
            if i % 1000 == 0:
                print(i)
        effsize = (slicesize-1) * dilation+1
        whichimg = torch.randint(0, len(mnist), (maxsize,))
        top = torch.randint(0, 28-effsize+1, (maxsize,))
        left = torch.randint(0, 28-effsize+1, (maxsize,))
        self.y = torch.stack([
            images[whichimg[i], top[i]:top[i]+effsize:dilation, left[i]:left[i]+effsize:dilation]
            for i in range(maxsize)
        ]).view(maxsize, -1)
        noise = torch.randn_like(self.y)
        interp = torch.rand((maxsize, 1))
        self.x = interp * self.y + (1-interp) * noise

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
