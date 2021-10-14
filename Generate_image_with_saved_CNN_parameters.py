#!/usr/bin/env python
# coding: utf-8
# Try sudo python3 Generate_image_with_saved_CNN_parameters.py, if the script could not be run properly after installing all the packages. Some computer have two GPU (one integrated with cpu, the other one is the independ GPU), in linux sometime the machine may not permission to use the GPU even the GPU is available. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
from skimage.filters import gaussian
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # tools to make zoomin or zoomout possible in the heatmap
import random
dtype = torch.float32 # The gpu used in this study can only supports float32
device = torch.device("cpu") # you can also use cuda:0 if cuda is availabe. 

plk = input("Please put the saved parameters here:" ) # BHgfp_3674td_03_cnn_250timesTrained.plk

# the network CNN used in this study
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
            nn.Sigmoid() # sigmoid is used in this study, because we want the output be at range of 0~1.    
        )
        
    def forward(self, x):
        x = self.out(x)
        return x

net = Net()
net.to(device)  

# read the images
sg = 1.0
td = io.imread("GFPMucD4td/C1-BHgfp_MucD4td_4days_03.tif") 
gfp = io.imread("GFPMucD4td/C2-BHgfp_MucD4td_4days_03.tif") 
af = io.imread("GFPMucD4td/C3-BHgfp_MucD4td_4days_03.tif") 
gfp = gaussian(gfp, sigma=sg)
td = gaussian(td, sigma=sg)
af = gaussian(af, sigma=sg)

autoFlu = gfp.mean() # will be used for non-bacterial pixels depletion in relative expression plotting

#load the trained parameter to the CNN
net.load_state_dict(torch.load(plk)) # load the pre-trained parameter to the CNN model

image = np.concatenate((td, gfp, af), dtype=np.float32) # concatenate three channels into one numpy array
image = image.reshape((3,512,512)) # reshape the input to 3*512*512
image = torch.tensor(image, dtype=dtype, device=device) # put the array into device, here is cpu. The data is transformed into Tensor. 
image = image.unsqueeze(0) # unsqueeze the data, while the pytorch can only uptake data as minibatch

AF = net(image).data.numpy() # predicted AF and transform it in to numpy array.
AF = AF.reshape((512,512)) # reshape the AF into a 512*512 image, or we could understant the output

bac_td = td - AF # baterical tdTomato should be the AF removed orignal tdTomato 

# make those image have the same colormap range. 
af[af>0.3]=0.3 #few pixels may be over-exposured. A max value should be invited in order to make all pixels in the proper range of a colormap.
AF[0][0]=af.max() # the range should be 0~0.3
AF[0][1]=0
td[0][0]=af.max()
td[0][1]=0

#########################################################################################
# show the predicted results in images. The new matplotlib package may give warning reports, just ignore them. 
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
ax = axes.ravel()

cmap = 'turbo'#'terrain'#'turbo' several other colormaps could also be used. 

im = ax[0].imshow(AF, cmap=cmap)
ax[0].set_title("network generated AFtd")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[1].imshow(td, cmap=cmap)
ax[1].set_title("tdTomato channel")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[2].imshow(gfp, cmap=cmap)
ax[2].set_title("GFP channel")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[3].imshow(af, cmap=cmap)
ax[3].set_title("AF channel")
divider = make_axes_locatable(ax[3])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[4].imshow(bac_td, cmap="binary")
ax[4].set_title("bacterial filter")
divider = make_axes_locatable(ax[4])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

bac_td[bac_td<0] = 0 # set the value<0 to 0. after subtraction, some pixels would be smaller than 0, which doesn't any sense and should be set to 0. 

im = ax[5].imshow(bac_td, cmap=cmap)
ax[5].set_title("tdTomato $-$ AFtd $>$ 0")
divider = make_axes_locatable(ax[5])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
plt.show()


##############################################################################
## relative expression. As there are pixels in GFP channel with value of 0, unwanted non-bacterial pixels should be depleted this time. 
bac_td = gaussian(bac_td, sigma=sg)
ratio = bac_td/gfp
ratio[ratio > 1.4] = 1.4
ratio[gfp < 0.6 * autoFlu] = 0
fig, axes = plt.subplots(1, 4, figsize=(18, 10), sharex=True, sharey=True)
ax = axes.ravel()

cmap = 'turbo'#'terrain'#'turbo'

im = ax[0].imshow(bac_td, cmap=cmap)
ax[0].set_title("computer generated bac_tdTomato")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[1].imshow(gfp, cmap=cmap)
ax[1].set_title("GFP channel")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)


im = ax[2].imshow(ratio, cmap=cmap)
ax[2].set_title("tdTomato/GFP")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[3].imshow(af, cmap="binary", alpha=0.4)
ratio[gfp < 0.6 * autoFlu] = None
im = ax[3].imshow(ratio, cmap=cmap)
ax[3].set_title("tdTomato/GFP")
divider = make_axes_locatable(ax[3])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
plt.show()
