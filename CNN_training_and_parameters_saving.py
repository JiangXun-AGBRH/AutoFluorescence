#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
from skimage.filters import gaussian
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # tools to make zoomin or zoomout possible in the heatmap
dtype = torch.float32
device = torch.device("cuda:0") # if cuda is not available, use device = torch.device("cpu") for training

roots = listdir("roots/")
bacteria = listdir("bacteria/")

# generate synthetic images of bacteria colonized roots according to Strategy II
def fake_im_generator(roots, bacteria):
    # all the images files in different channels
    root_td = sorted([f'roots/{i}' for i in roots if "C1" in i])
    root_gfp = sorted([f'roots/{i}' for i in roots if "C2" in i])
    root_af = sorted([f'roots/{i}' for i in roots if "C3" in i])
    bac_td = sorted([f'bacteria/{i}' for i in bacteria if "C1" in i])
    bac_gfp = sorted([f'bacteria/{i}' for i in bacteria if "C2" in i])
    sg = 1.0
    for i in range(len(root_af)):
        for j in range(len(bac_td)):
            #read all the images files in different channels as nparray
            td_root = io.imread(root_td[i])
            td_root = gaussian(td_root, sigma=sg)
            gfp_root = io.imread(root_gfp[i])
            gfp_root = gaussian(gfp_root, sigma=sg)
            af_root = io.imread(root_af[i])
            af_root = gaussian(af_root, sigma=sg)
            td_bac = io.imread(bac_td[j])
            td_bac = gaussian(td_bac, sigma=sg)
            gfp_bac = io.imread(bac_gfp[j])
            gfp_bac = gaussian(gfp_bac,sigma=sg)
            
            tdTomato = td_root + td_bac
            gfp = gfp_root + gfp_bac
            td_root = np.array(td_root, dtype=np.float32)
            td_root = td_root.reshape((1,1,512,512))
            td_root = torch.tensor(td_root, dtype=dtype, device=device)
            
            image = np.concatenate((tdTomato, gfp, af_root), dtype=np.float32)
            image = image.reshape((3,512,512))
            image = torch.tensor(image, dtype=dtype, device=device)
            image = image.unsqueeze(0)
            
            yield image, td_root


# the network CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(3,12,5),
            nn.ReLU(),
            
            nn.Conv2d(12,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32,96,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(96,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128,256,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.ConvTranspose2d(256, 128, 5, stride = 2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 96, 5, stride = 2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(96, 32, 5, stride = 2, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 8, 5, stride = 2, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8,1,4),
            nn.Sigmoid() #output of sigmod is at 0～1, tiff or zen files could not be used directly, because Int type could not be used for machine learning
        )
        
    def forward(self, x):
        x = self.out(x)
        return x

net = Net()
net.to(device)   

# read the image and prepare the input as described in Strategy I
sg = 1.0
td = io.imread("C1-azo2848proPMP 63.tif")
gfp = io.imread("C2-azo2848proPMP 63.tif")
af = io.imread("C3-azo2848proPMP 63.tif")

gfp = gaussian(gfp, sigma=sg)
td = gaussian(td, sigma=sg)
af = gaussian(af, sigma=sg)

factor = 1.4 # set the factor manually. only the bacteria should be filtered. 
autoFlu = gfp.mean()
bacteria_filter = gfp > autoFlu * factor

gfp[bacteria_filter] = 0 # GFP without signal of bacteria

td[bacteria_filter] = 0 # tdTomato without signal of bacteria

af[bacteria_filter] = 0 # AF without signal of bacteria

# check the bacteria-depletion-filter, if the filter is not good, change the factor. 
fig, ax = plt.subplots(3,1, figsize=(14,18), dpi= 90, facecolor='w', edgecolor='k')
ax[0].imshow(td, cmap='binary')
ax[0].set_title("tdTomato")

ax[1].imshow(gfp, cmap='binary')
ax[1].set_title("GFP")

ax[2].imshow(af, cmap='binary')
ax[2].set_title("AF")

fig.tight_layout()
plt.show()

# combine all the channels together to form an 3*512*512 image
fig = np.concatenate((td, gfp, af), dtype=np.float32)
fig = torch.tensor(fig, device=device, dtype=dtype)
fig = fig.reshape((1,3,512,512))

td = torch.tensor(np.array(td, dtype=np.float32), device=device, dtype=dtype)
td = td.reshape((1,512,512))
td = td.unsqueeze(1)
print(td.shape, fig.shape)

# hyperparameters could be changed. In this study learning_rate was suggest to be at 1e-3 to 1e-5. If the MSELoss could not be reduced but still greater than 0.1, 1e-4 or 1e-5 could be tried. 
# if you want to use a pre-trained Model for further training, please set the learning rate to 1e-4 or 1e-5.
loss_fn = torch.nn.MSELoss(reduction='sum') #here is the sum of MSELoss
learning_rate = 1e-3 # if it is trained from scratch, use 1e-3 to save time. 
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# training start here
#net.load_state_dict(torch.load("cnn_saved_parameters01.plk")) # uncomment this if you want to use a pre-trained Model.
for t in range(20): # It is recommented to save the parameters each 50~200 cycles. If the output is not good enough, reload the saved Model and start training again, and save the Model in another file. Do this until the MSELoss is smaller than 0.05 or even lower. 
    for image, label in fake_im_generator(roots, bacteria):
        # training with Strategy II
        y = net(image) 
        loss = loss_fn(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # training with Strategy I
        y = net(fig)
        loss = loss_fn(y, td)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if t % 10 == 0:
        print(f"{t} \t {loss.item()}")

# uncomment this, if you want the training to be stopped and save the trained parameters when the SUM of MSELoss < 0.05 automatically
"""        
    if loss.item() < 0.05:
        torch.save(net.state_dict(),"cnn_saved_parameters%d.plk" %(t)) 
        break
"""

torch.save(net.state_dict(),"cnn_saved_parameters01.plk")  
