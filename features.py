import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.distributions import Beta

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
        images = torch.zeros((len(dataset), imgsize, imgsize))
        i = 0
        for image, label in dataset:
            images[i] = image.squeeze()
            i += 1
            if i % 1000 == 0:
                #plt.imshow(images[i-1].numpy())
                #plt.colorbar()
                #plt.show()
                print(i)
        
        effsize = (kernel-1) * dilation + 1
        padding = effsize // 2
        padimages = F.pad(images,(padding, padding, padding, padding))
        
        whichimg = torch.randint(0, len(dataset), (maxsize,))
        y = torch.randint(0, imgsize, (maxsize,))
        x = torch.randint(0, imgsize, (maxsize,))
        
        self.y = images[whichimg, y, x].unsqueeze(1)
        drawnslices = torch.stack([
            padimages[whichimg[i], y[i]:y[i]+2*padding+1:dilation, x[i]:x[i]+2*padding+1:dilation]
            for i in range(maxsize)
        ]).reshape(maxsize, kernel*kernel)
        noise = torch.randn_like(drawnslices)
        beta = Beta(2., 1.)
        interp = beta.sample((maxsize, 1))
        self.x = interp * drawnslices + (1-interp) * noise
        
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
