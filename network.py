#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MuyGP(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.trainX = None
        self.trainy = None
        self.ymean = nn.Linear(inDim, outDim)
        self.l = nn.Parameter(torch.tensor(15.))
        #self.a = 1.
        self.a = nn.Parameter(torch.tensor(0.5))
        self.nn = 128

    def kernel(self, A, B):
        d = torch.cdist(A, B)
        val = self.a * torch.exp(-(d ** 2) / (2. * self.l ** 2))
        #val = self.a * (1 + np.sqrt(3) * d / self.l) * torch.exp(-np.sqrt(3) * d / self.l)
        #val = self.a * torch.exp(-d / self.l)
        return val

    def forward(self, x):
        #ymean = self.ymean(x).unsqueeze(1)
        ymean = 0.
        dists = torch.cdist(x, self.trainX)
        if self.training:
            _, neighbors = torch.topk(dists, self.nn+1, largest=False, dim=1)
            nX = self.trainX[neighbors[:,1:]]
            ny = self.trainy[neighbors[:,1:]]
        else:
            _, neighbors = torch.topk(dists, self.nn, largest=False, dim=1)
            nX = self.trainX[neighbors]
            ny = self.trainy[neighbors]
            print(_.min())
            #plt.imshow(x[0,:].view(28, 28).detach().cpu().numpy())
            #plt.show()
        ny = ny - ymean
        auto = self.kernel(nX, nX)
        #print(auto)
        autoCov = torch.linalg.inv(auto)
        crossCov = self.kernel(x.unsqueeze(1), nX)
        kWeights = crossCov @ autoCov
        y = kWeights @ ny
        yVar = self.a * torch.ones(x.size(0), device=x.device) - \
            (kWeights @ crossCov.transpose(1, 2)).squeeze()
        return (y + ymean).squeeze(), yVar


class NN(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.l = 1.
        self.a = 1.
        self.fcnn = nn.Sequential(
            nn.Linear(inDim, outDim),
            nn.LeakyReLU(),
            nn.Linear(outDim, outDim)
        )

    def forward(self, x):
        x = self.fcnn(x)
        return x, torch.ones_like(x)
    
