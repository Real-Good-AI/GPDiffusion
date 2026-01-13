#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from features import FlowDataset
from network import MuyGP, NN
from torch.utils.data import DataLoader

timesteps = 100
kernel = 13 #MUST be odd, otherwise there's no center pixel
imgsize = 64
nimg = 4
dilation = 1

def trainModel(loader, gp, device):
    vdata = FlowDataset(t=timesteps, maxsize=10, train=False, kernel=kernel, dilation=dilation)
    vloader = DataLoader(vdata, batch_size=512, pin_memory=True)
    
    epoch = 0
    epochLoss = []
    validsLoss = []
    gpopt = optim.AdamW(gp.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(gpopt, patience=0, cooldown=4)
    
    while gpopt.param_groups[0]["lr"] > 1e-7 and epoch < 10:
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
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FlowDataset(t=timesteps, maxsize=100, train=True, kernel=kernel, dilation=dilation)
    loader = DataLoader(data, batch_size=2048, shuffle=True, pin_memory=True)
    
    gp = MuyGP(kernel*kernel, kernel*kernel).to(device)
    gp.trainX = data.x.to(device)
    gp.trainy = data.y.to(device)
    gp.ymean = gp.trainy.mean(dim=0, keepdim=True)
    #gp = NN(784, 784).to(device)

    trainModel(loader, gp, device)
    
    with torch.no_grad():
        gp.eval()
        test = torch.randn((nimg, imgsize, imgsize), device=device)
        temp = test.clone()
        pos = torch.cartesian_prod(torch.arange(imgsize), torch.arange(imgsize)).to(device)
        pos = 2*pos / imgsize - 1
        #print(pos)
        for t in range(timesteps):
            convs = F.unfold(temp.unsqueeze(1), kernel_size=kernel, dilation=dilation, padding=((kernel-1) * dilation+1) // 2)
            convs = torch.hstack((convs.transpose(1, 2).reshape(-1, kernel*kernel), 5*pos.repeat(nimg, 1)))
            convsout, var = gp(convs)
            var = var.reshape(nimg, imgsize, imgsize)
            out = convsout.reshape(nimg, imgsize, imgsize)
            combo = np.sin((t+1)/timesteps * np.pi/2)
            temp = (1-combo) * test + combo * out
            
            if t % 10 == 0:
                print(t)
                plt.imshow(temp[0].detach().cpu().numpy())
                plt.colorbar()
                plt.show()
            
        test = temp
        for i in range(test.size(0)):
            img = test[i].view(imgsize, imgsize).detach().cpu().numpy()
            plt.imshow(img)
            plt.show()

        
