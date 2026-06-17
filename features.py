import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.distributions import Beta
from random import shuffle

class FlowDataset(Dataset):
    def __init__(self, t=10, maxsize=2000, kernel=5, train=True, dilation=1):
        imgsize = 28
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        
        '''
        if train:
            keyword = "train"
        else:
            keyword = "test"
        transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        '''
        dataset = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
        #dataset = torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
        #dataset = torchvision.datasets.SVHN(root="./data", split=keyword, download=True, transform=transform)
        images = torch.zeros((maxsize, imgsize, imgsize))
        i = 0
        for image, label in dataset:
            images[i] = image.squeeze()
            i += 1
            if i >= maxsize:
                break
            if i % 10000 == 0:
                plt.axis("off")
                plt.imshow(images[i-1].numpy(), cmap="gray")
                #plt.colorbar()
                plt.show()
                print(i)
        
        effsize = (kernel-1) * dilation + 1
        padding = effsize // 2
        padimages = F.pad(images,(padding, padding, padding, padding), value=-1.)
        
        idx = torch.cartesian_prod(torch.arange(maxsize), torch.arange(imgsize), torch.arange(imgsize))
        drawnslices = torch.stack([
            padimages[idx[i,0], idx[i,1]:idx[i,1]+effsize:dilation, idx[i,2]:idx[i,2]+effsize:dilation]
            for i in range(idx.size(0))
        ]).reshape(maxsize*imgsize*imgsize, kernel*kernel)
        pos = (2*idx[:,1:]/imgsize - 1.)
        drawnslices = torch.hstack((drawnslices, 5*pos))
        drawnslices = torch.unique(drawnslices, dim=0)
        self.y = drawnslices[:, (kernel*kernel-1) // 2].unsqueeze(-1)
        print(self.y.size())
        noise = torch.randn_like(drawnslices[:,:-2])
        if train:
            beta = Beta(1., 1.)
        else:
            beta = Beta(1., 1.)
        interp = beta.sample((drawnslices.size(0), 1))
        self.denoise = drawnslices
        self.x = torch.hstack((torch.sqrt(interp) * drawnslices[:,:-2] + torch.sqrt(1-interp) * noise, drawnslices[:,-2:]))
        print(self.x.size())
        #for thing in self.x[torch.randperm(self.x.size(0))]:
            #plt.axis("off")
            #plt.imshow(thing[:-2].view(kernel, kernel), cmap="gray")
            #plt.show()
        
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
