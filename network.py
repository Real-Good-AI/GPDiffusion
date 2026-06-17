import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MuyGP(nn.Module):
    def __init__(self, inDim):
        super().__init__()
        self.trainX = None
        self.trainy = None
        self.ymean = None
        #self.l = nn.Parameter(torch.zeros((1, inDim)))
        self.a = torch.tensor(0.)
        #self.a = nn.Parameter(torch.tensor(0.))
        self.t = nn.Parameter(torch.tensor(0.))
        self.l = nn.Parameter(torch.tensor(0.))
        self.nn = 128

    def kernel(self, A, B):
        l = torch.exp(self.l)
        A = A / l
        B = B / l
        d = torch.cdist(A, B)
        d = d / np.sqrt(A.size(-1))
        #val = torch.exp(-d)
        val = torch.exp(-(d ** 2)/2)
        return val

    def forward(self, x):
        l = torch.exp(self.l)
        t = torch.exp(self.t)
        ymean = self.ymean
        dists = torch.cdist(x/l, self.trainX/l)
        if self.training:
            _, neighbors = torch.topk(dists, self.nn+1, largest=False, dim=1)
            nX = self.trainX[neighbors[:,1:]]
            ny = self.trainy[neighbors[:,1:]] - ymean
        else:
            _, neighbors = torch.topk(dists, self.nn, largest=False, dim=1)
            nX = self.trainX[neighbors]
            ny = self.trainy[neighbors] - ymean
        auto = self.kernel(nX, nX) + t * torch.eye(self.nn, device=nX.device).unsqueeze(0)
        autoCov = torch.linalg.inv(auto)
        crossCov = self.kernel(x.unsqueeze(1), nX)
        kWeights = crossCov @ autoCov
        a = (ny.transpose(-2, -1) @ autoCov @ ny).mean()
        self.a = torch.log(a)
        #a = torch.exp(self.a)
        y = kWeights @ ny
        yVar = self.kernel(x.unsqueeze(1), x.unsqueeze(1)) - \
            (kWeights @ crossCov.transpose(1, 2))
        return (y + ymean).squeeze(1), a * torch.clamp(yVar.squeeze(), min=1e-10)

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
    
