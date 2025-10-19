#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from features import DiffDataset
from network import MuyGP
from torch.utils.data import DataLoader

timesteps = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DiffDataset(t=timesteps, maxsize=1000, train=True)
    loader = DataLoader(data, batch_size=1024, shuffle=True, pin_memory=True)

    gp = MuyGP(784, 784).to(device)
    gp.trainX = data.x.to(device)
    gp.trainy = data.y.to(device)

    vdata = DiffDataset(t=timesteps, maxsize=100, train=False)
    vloader = DataLoader(vdata, batch_size=4096, pin_memory=True)
    
    epoch = 0
    epochLoss = []
    validsLoss = []
    gpopt = optim.Adam(gp.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(gpopt, cooldown=4)

    while gpopt.param_groups[0]["lr"] > 1e-4 and epoch < 20:
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
        scheduler.step(runningLoss)
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
        test = torch.randn((10, 784), device=device)
        for t in range(timesteps):
            test, var = gp(test)
            img = test[0].view(28, 28).detach().cpu().numpy()
            plt.imshow(img)
            plt.show()
