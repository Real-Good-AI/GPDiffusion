#!/home/ewbell/miniforge3/envs/gpdiff/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from features import FlowDataset
from network import MuyGP, NN
from torch.utils.data import DataLoader

timesteps = 100
kernel = 8
imgsize = 64
nimg = 2
dilation = 2
stride = 1

def trainModel(loader, gp, device):
    vdata = FlowDataset(t=timesteps, maxsize=10000, train=False, slicesize=kernel, dilation=dilation)
    vloader = DataLoader(vdata, batch_size=512, pin_memory=True)
    
    epoch = 0
    epochLoss = []
    validsLoss = []
    gpopt = optim.AdamW(gp.parameters(), lr=1e-1, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(gpopt, patience=2, cooldown=4)
    
    while gpopt.param_groups[0]["lr"] > 1e-4 and epoch < 5:
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
    data = FlowDataset(t=timesteps, maxsize=100000, train=True, slicesize=kernel, dilation=dilation)
    loader = DataLoader(data, batch_size=512, shuffle=True, pin_memory=True)
    
    gp = MuyGP(kernel*kernel, kernel*kernel).to(device)
    gp.trainX = data.x.to(device)
    gp.trainy = data.y.to(device)
    gp.ymean = gp.trainy.mean(dim=0, keepdim=True)
    #gp = NN(784, 784).to(device)

    trainModel(loader, gp, device)
    
    with torch.no_grad():
        gp.eval()
        fold = nn.Fold(output_size=imgsize, kernel_size=kernel, stride=stride, dilation=dilation)
        unfold = nn.Unfold(kernel_size=kernel, stride=stride, dilation=dilation)
        norm = fold(unfold(torch.ones((1, 1, imgsize, imgsize), device=device)))
        test = torch.randn((nimg, imgsize, imgsize), device=device)
        temp = test 
        for t in range(timesteps):
            convs = unfold(temp.unsqueeze(1)).transpose(1, 2).reshape(-1, kernel*kernel)
            convsout, var = gp(convs)
            out = fold(convsout.reshape(nimg, -1, kernel*kernel).transpose(1, 2)) / norm
            out = out.squeeze()
            temp = (timesteps-t-1)/timesteps * test + (t+1)/timesteps * out
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

        
