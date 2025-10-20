#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from features import DiffDataset, makeAlphaBar
from network import MuyGP, NN
from torch.utils.data import DataLoader

timesteps = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DiffDataset(t=timesteps, maxsize=100000, train=True)
    loader = DataLoader(data, batch_size=512, shuffle=True, pin_memory=True)
    
    gp = MuyGP(785, 784).to(device)
    gp.trainX = data.x.to(device)
    gp.trainy = data.y.to(device)
    
    #gp = NN(785, 784).to(device)
    vdata = DiffDataset(t=timesteps, maxsize=10000, train=False)
    vloader = DataLoader(vdata, batch_size=512, pin_memory=True)
    
    epoch = 0
    epochLoss = []
    validsLoss = []
    gpopt = optim.Adam(gp.parameters(), lr=1e-2)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(gpopt, cooldown=4)
    scheduler = optim.lr_scheduler.ExponentialLR(gpopt, gamma=0.8912)
    while gpopt.param_groups[0]["lr"] > 1e-4 and epoch < 100:
        print(gpopt.param_groups[0]["lr"])
        runningLoss = 0.
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            gpopt.zero_grad()
            output, var = gp(x)
            var = torch.clamp(var, min=1e-10)
            errors = (output - y) ** 2. / var.unsqueeze(1)
            loss = errors.sum() + y.size(1) * torch.log(var).sum()
            loss.backward()
            gpopt.step()
            runningLoss += loss.item()
        epochLoss.append(runningLoss)
        scheduler.step()
        epoch += 1
        with torch.no_grad():
            gp.eval()
            validLoss = 0.
            for x, y in vloader:
                x = x.to(device)
                y = y.to(device)
                output, var = gp(x)
                var = torch.clamp(var, min=1e-10)
                errors = (output - y) ** 2. #/ var.unsqueeze(1)
                loss = errors.sum() #+ y.size(1) * torch.log(var).sum()
                validLoss += loss.item()
            validsLoss.append(validLoss)
            gp.train()
        print(epoch, epochLoss[-1], validsLoss[-1])
        print(gp.a)
        print(gp.l)
    
    with torch.no_grad():
        gp.eval()
        test = torch.randn((3, 784), device=device)
        steps, alpha, abar = makeAlphaBar(timesteps)
        steps = steps.to(device)
        alpha = alpha.to(device)
        abar = abar.to(device)
        for t in reversed(range(timesteps)):
            eps, var = gp(torch.hstack((test, steps[t].unsqueeze(0).expand(test.size(0),1))))
            test = 1/torch.sqrt(alpha[t]) * (test - (1-alpha[t])/torch.sqrt(1-abar[t]) * eps)
            print(test)
        for i in range(test.size(0)):
            img = test[i].view(28, 28).detach().cpu().numpy()
            plt.imshow(img)
            plt.show()
        
