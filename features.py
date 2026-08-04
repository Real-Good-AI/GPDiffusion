import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from random import shuffle
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

class FlowDataset(Dataset):
    def __init__(self, t=10, maxsize=2000, kernel=5, train=True, dilation=1, channels=3, padval=0.):
            
        imgcap = 10000
        
        imgsize = 28
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        
        dataset = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
        #dataset = torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
        
        '''
        imgsize=32
        if train:
            keyword = "train"
        else:
            keyword = "test"
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        
        dataset = torchvision.datasets.SVHN(root="./data", split=keyword, download=True, transform=transform)
        '''
        drawninds = torch.randperm(len(dataset))[:imgcap]
        images = torch.zeros((drawninds.size(0), channels, imgsize, imgsize))
        for i in range(drawninds.size(0)):
            ind = drawninds[i]
            image, _ = dataset[ind]
            images[i] = image.view(-1, imgsize, imgsize)
        self.imgavg = images.mean(dim=0)
        effsize = (kernel-1) * dilation + 1
        padding = effsize // 2
        padimages = F.pad(images,(padding, padding, padding, padding), value=padval)
        drawnslices = F.unfold(padimages, kernel_size=kernel, dilation=dilation).transpose(-1,-2)
        idx = torch.cartesian_prod(torch.arange(imgsize), torch.arange(imgsize))
        pos = (2 * idx / (imgsize-1) - 1.).expand(drawnslices.size(0), -1, -1)
        drawnslices = torch.cat((drawnslices, pos), dim=-1)
        drawnslices = drawnslices.reshape(images.size(0)*imgsize*imgsize, kernel*kernel*channels+2)
        dsnp = drawnslices.numpy()
        kmeans = MiniBatchKMeans(n_clusters=maxsize, verbose=10, max_iter=1000, batch_size=4096).fit(dsnp)
        centroids = kmeans.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(centroids, dsnp)
        drawnslices = drawnslices[torch.tensor(closest)]
        self.x = drawnslices
        self.y = drawnslices[:,:-2].view(-1, channels, kernel, kernel)[:, :, kernel//2, kernel//2]
        print(self.x.size())
        print(self.y.size())
        
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
