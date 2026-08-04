import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanGP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.trainX = None
        self.trainy = None
        self.l = nn.Parameter(torch.tensor(0.))
        self.nn = 8
        
    def kernel(self, A, B):
        d = torch.cdist(A, B)
        val = torch.exp(-d)
        return val

    def forward(self, x):
        l = torch.exp(self.l)
        dists = torch.cdist(x/l, self.trainX/l)
        _, neighbors = torch.topk(dists, self.nn, largest=False, dim=1)
        nX = self.trainX[neighbors]
        ny = self.trainy[neighbors]
        auto = self.kernel(nX/l, nX/l) + 1e-4 * torch.eye(self.nn, device=nX.device).unsqueeze(0)
        autoCov = torch.linalg.inv(auto)
        crossCov = self.kernel((x/l).unsqueeze(1), nX/l)
        kWeights = crossCov @ autoCov
        y = kWeights @ ny
        return y
        
class MuyGP(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.trainX = None
        self.trainstd = None
        self.trainy = None
        
        self.ymean = MeanGP(outDim)
        self.a = torch.tensor(0.)
        self.t = nn.Linear(1, 1)
        self.l = nn.Linear(1, 3)
        self.nn = 32

    def kernel(self, A, B):
        d = torch.cdist(A, B)
        #val = torch.exp(-d)
        val = torch.exp(-(d ** 2)/2)
        return val

    def forward(self, x):
        ymean = self.ymean(x[:,-2:])
        std = torch.std(x[:,:-2], dim=-1, keepdim=True)
        l = torch.exp(self.l(std))
        l = torch.hstack((l[:,0].unsqueeze(-1).repeat(1,x.size(1)-2), \
                          l[:,1].unsqueeze(-1), l[:,2].unsqueeze(-1)))
        trainl = torch.exp(self.l(self.trainstd))
        trainl = torch.hstack((trainl[:,0].unsqueeze(-1).repeat(1,x.size(1)-2), \
                               trainl[:,1].unsqueeze(-1), trainl[:,2].unsqueeze(-1)))
        t = torch.exp(self.t(std)).unsqueeze(-1)
        dists = torch.cdist(x/l, self.trainX/trainl)
        if self.training:
            _, neighbors = torch.topk(dists, self.nn+1, largest=False, dim=1)
            nX = self.trainX[neighbors[:,1:]]
            nl = trainl[neighbors[:,1:]]
            ny = self.trainy[neighbors[:,1:]] - ymean            
        else:
            _, neighbors = torch.topk(dists, self.nn, largest=False, dim=1)
            nX = self.trainX[neighbors]
            nl = trainl[neighbors]
            ny = self.trainy[neighbors] - ymean
        auto = self.kernel(nX/nl, nX/nl) + t * torch.eye(self.nn, device=nX.device).unsqueeze(0)
        autoCov = torch.linalg.inv(auto)
        crossCov = self.kernel((x/l).unsqueeze(1), nX/nl)
        kWeights = crossCov @ autoCov
        a = (ny.transpose(-2, -1) @ autoCov @ ny).mean()
        self.a = torch.log(a)
        y = kWeights @ ny
        yVar = self.kernel((x/l).unsqueeze(1), (x/l).unsqueeze(1)) - \
            (kWeights @ crossCov.transpose(1, 2))
        return (y + ymean).squeeze(1), a * torch.clamp(yVar.squeeze(), min=1e-10)

class NN(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.l = nn.Linear(1, 1)
        self.a = torch.tensor(1.)
        self.t = nn.Linear(1, 1)
        self.fcnn = nn.Sequential(
            nn.Linear(inDim, outDim),
            nn.LeakyReLU(),
            nn.Linear(outDim, outDim)
        )

    def forward(self, x):
        x = self.fcnn(x)
        return x, torch.ones(x.size(0), device=x.device)
    
