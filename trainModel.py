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

#Feel free to change these parameters! Just keep VRAM in mind
timesteps = 100 #Timesteps for diffusion
kernel = 11 #Convolutional kernel size, MUST be odd, otherwise there's no center pixel
imgsize = 90 #Size of canvas for image generation
nimg = 1 #Number of images to generate
dilation = 1 #Dilation of convolutional kernel
channels = 1 #Number of channels in dataset (1 for greyscale, 3 for RGB color)
padval = -1. #Value for image padding (-1 for black background images like MNIST, otherwise 0)

def trainModel(data, gp, device):
    loader = DataLoader(data, batch_size=2048, shuffle=True, pin_memory=True)
    vdata = FlowDataset(t=timesteps, maxsize=1024, train=False, kernel=kernel, dilation=dilation, channels=channels)
    vloader = DataLoader(vdata, batch_size=512, pin_memory=True)
    
    epoch = 0
    epochLoss = []
    validsLoss = []
    gpopt = optim.AdamW(gp.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(gpopt, patience=4, cooldown=4)
    
    while gpopt.param_groups[0]["lr"] > 1e-5:
        print(gpopt.param_groups[0]["lr"])
        runningLoss = 0.
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            gpopt.zero_grad()
            coef = torch.rand((x.size(0), 1), device=device)
            noise = torch.randn_like(x[:,:-2])
            x = torch.hstack((coef * x[:, :-2] + (1-coef) * noise, x[:, -2:]))
            output, var = gp(x)
            errors = (output - y) ** 2. / var.unsqueeze(-1)
            ldiff = gp.l(torch.tensor([[1.],[0.1]], device=device))
            penalty = 1000000.*torch.clamp(ldiff[1,0] - ldiff[0,0], min=0.) #Penalty is to keep l high for high noise
            loss = errors.sum() + y.size(1) * torch.log(var).sum() + penalty
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
                errors = (output - y) ** 2. / var.unsqueeze(-1)
                loss = errors.sum() + y.size(1) * torch.log(var).sum()
                validLoss += loss.item()
            validsLoss.append(validLoss)
            gp.train()
        print(epoch, epochLoss[-1], validsLoss[-1])
        example = torch.tensor([[0.1],[0.5],[1.]], device=device)
        print("a: "+str(torch.exp(gp.a)))
        print("t: "+str(torch.exp(gp.t(example))))
        print("l: "+str(torch.exp(gp.l(example))))
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = FlowDataset(t=timesteps, maxsize=2**13, train=True, kernel=kernel, dilation=dilation, channels=channels, padval=padval)
    
    gp = MuyGP(kernel*kernel*channels+2, channels).to(device)
    gp.trainX = data.x.to(device)
    gp.trainstd = torch.std(gp.trainX[:,:-2], dim=-1, keepdim=True)
    gp.trainy = data.y.to(device)
    pos = torch.cartesian_prod(torch.arange(data.imgavg.size(1)), torch.arange(data.imgavg.size(2))).to(device)
    pos = 2*pos / (data.imgavg.size(1)-1) - 1
    gp.ymean.trainX = pos + 1e-4 * torch.randn_like(pos) #noise is to keep cdist from having duplicates
    gp.ymean.trainy = torch.flatten(data.imgavg.to(device), start_dim=1).transpose(0,1)

    #gp = NN(kernel*kernel*channels+2, channels).to(device)

    trainModel(data, gp, device)
    with torch.no_grad():
        gp.eval()
        pad = ((kernel-1) * dilation + 1) // 2
        test = torch.randn((nimg, channels, imgsize+2*pad, imgsize+2*pad), device=device)
        test = test + F.pad(torch.zeros((nimg, channels, imgsize, imgsize), device=device), (pad,pad,pad,pad), value=padval)
        temp = test.clone()
        pos = torch.cartesian_prod(torch.arange(imgsize), torch.arange(imgsize)).to(device)
        pos = 2*pos / (imgsize-1) - 1
        for t in range(timesteps):
            convs = F.unfold(temp, kernel_size=kernel, dilation=dilation).transpose(1, 2)
            convs = torch.cat((convs, pos.expand(nimg, -1, -1)), dim=-1)
            convs = convs.reshape(-1, kernel*kernel*channels+2)
            convsout, var = gp(convs)
            var = var.reshape(nimg, imgsize, imgsize)
            out = convsout.reshape(nimg, imgsize, imgsize, channels).permute(0, 3, 1, 2)
            out = F.pad(out, (pad, pad, pad, pad), value=padval)
            offset = 0.008
            abar1 = np.sin((t/timesteps + offset) / (1 + offset) * np.pi/2) ** 2
            abar2 = np.sin(((t+1)/timesteps + offset) / (1 + offset) * np.pi/2) ** 2
            eps = (temp - np.sqrt(abar1) * out) / np.sqrt(1 - abar1)
            print(np.sqrt(abar1), np.sqrt(abar2))
            temp = np.sqrt(abar2) * out + np.sqrt(1 - abar2) * eps
            
            if t % 10 == 0:
                print(t)
                plt.axis("off")
                plt.imshow((temp[0].detach().cpu().permute(1,2,0).numpy()+1)/2, vmin=0., vmax=1.)
                plt.colorbar()
                plt.show()
                plt.axis("off")
                plt.imshow(var[0].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.show()
                plt.axis("off")
                plt.imshow((out[0].detach().cpu().permute(1,2,0).numpy()+1)/2, vmin=0., vmax=1.)
                plt.colorbar()
                plt.show()
            
        test = temp
        for i in range(test.size(0)):
            plt.axis("off")
            plt.imshow((test[i,:,pad:-pad,pad:-pad].detach().cpu().permute(1,2,0).numpy()+1)/2, vmin=0., vmax=1.)
            plt.colorbar()
            plt.show()

        
