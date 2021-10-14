#!/usr/bin/env python
# coding: utf-8
# the saved model/parameters were all from CUDA training, if you want to run the script with the plk file from the supplement, please make sure CUDA is available in your python!!! 

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
dtype = torch.float32
device = torch.device("cpu")

plk = input("Please put the saved parameters here:" ) # use the trained Model parameters

# generate synthetic images for accuracy test
def gfpTd_generator(image, factor): # for example: "-image01.tif", tdTomato = gfp * factor, sigma for gaussin is 1.0
    sg = 1.0
    tdTomato = "BHgfp_3channels/C1" + image
    gfp = "BHgfp_3channels/C2" + image
    af = "BHgfp_3channels/C3" + image
    tdTomato = io.imread(tdTomato)
    tdTomato = gaussian(tdTomato, sigma=sg)
    gfp = io.imread(gfp)
    gfp = gaussian(gfp, sigma=sg)
    af = io.imread(af)
    af = gaussian(af, sigma=sg)
    filter_gfp = (gfp > 1.8 * gfp.mean())
    filteration = gfp.copy()
    filteration[filter_gfp] = 1
    filteration[filteration<1] = 0
    
    td = tdTomato + factor * gfp * filteration
    td = np.array(td, dtype=np.float32)
    #td = td_root.reshape((1,1,512,512))
    #td = torch.tensor(td, dtype=dtype, device=device)
    
    im = np.concatenate((td, gfp, af), dtype=np.float32)
    im = im.reshape((3,512,512))
    #im = torch.tensor(im, dtype=dtype, device=device)
    #im = image.unsqueeze(0)
    return im, tdTomato


# the CNN network
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
            nn.Sigmoid() #Tanh is at-1～1, sigmod is at 0～1, if use tif file for training please change     
        )
        
    def forward(self, x):
        x = self.out(x)
        return x

net = Net()  

# if you want to test other images, change the arguments here
image, tdTomato = gfpTd_generator("-image01.tif", 0.23)
td, gfp, af = image[0], image[1], image[2]
factor = 1.8 # set the factor manually 
autoFlu = gfp.mean()
bacteria_filter = gfp > autoFlu * factor

plt.imshow(bacteria_filter) # check whether the bacteria-depletion-filter is precise
plt.show()

# load the parameters to the CNN network
net.load_state_dict(torch.load(plk))

# generate the input  for processing, # if you want to test other images, change the arguments here
image, tdTomato = gfpTd_generator("-image01.tif", 0.23)
td, gfp, af = image[0], image[1], image[2]
image = np.concatenate((td, gfp, af), dtype=np.float32)
image = image.reshape((3,512,512)) 
image = torch.tensor(image, dtype=dtype, device=device)
image = image.unsqueeze(0)

AF = net(image).data.numpy() # transform the output Tensor to numpy array. 
AF = AF.reshape((512,512)) # reshape the output to a 512*512 image/array.
bac_td = td - AF # baterical tdTomato should be the AF removed orignal tdTomato 
bac_td0 = td - tdTomato # the expected artificial bacterial tdTomato
af[af>0.4]=0.4 # set the max for af to have a better view of the heatmap
tdTomato[0][0]=af.max() # To compare different images, set the colormap to the same range.
AF[0][0]=af.max() # To compare different images, set the colormap to the same range.
tdTomato[0][1]=0 # To compare different images, set the colormap to the same range.
AF[0][1]=0 # To compare different images, set the colormap to the same range.
td[0][0]=af.max() # To compare different images, set the colormap to the same range.
td[0][1]=0 # To compare different images, set the colormap to the same range.

#########################################################################################
# show the predicted results in images. The new matplotlib package may give warning reports, just ignore them. 
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
ax = axes.ravel()

cmap = 'turbo'#'terrain'#'turbo'

im = ax[0].imshow(AF, cmap=cmap)
ax[0].set_title("network generated AFtd")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[1].imshow(td, cmap=cmap)
ax[1].set_title("tdTomato + bac-tdTomato")
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

im = ax[4].imshow(bac_td0, cmap=cmap)
ax[4].set_title("bac-tdTomato")
divider = make_axes_locatable(ax[4])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

bac_td[bac_td<0] = 0
bac_td[bac_td>0.007] = 0.007
bac_td[0][0] = 0.007

im = ax[5].imshow(bac_td, cmap=cmap)
ax[5].set_title("tdTomato $-$ AFtd $>$ 0")
divider = make_axes_locatable(ax[5])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
plt.show()


# In[9]:


fig, ax = plt.subplots(1,3, figsize=(16,6), facecolor='w', edgecolor='k')

xmin = 0
xmax = 0.014
ymin = 0
ymax = 0.014

im = ax[0].hexbin(tdTomato, tdTomato, gridsize=120, bins='log', cmap='turbo')
ax[0].set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax[0].set_title("original_tdTomato to original_tdTomato")
ax[0].set_xlabel("value of pixels in original_tdTomato")
ax[0].set_ylabel("value of pixels in original_tdTomato")

im = ax[1].hexbin(tdTomato, td, gridsize=100, bins='log', cmap='turbo')
ax[1].set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax[1].set_title("tdTomato_with_bacteria to original_tdTomato")
ax[1].set_xlabel("value of pixels in original_tdTomato")
ax[1].set_ylabel("value of pixels in tdTomato_with_bacteria")

im = ax[2].hexbin(tdTomato, AF, gridsize=100, bins='log', cmap='turbo')
ax[2].set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax[2].set_title("network_generated_AFtd to original_tdTomato")
ax[2].set_xlabel("value of pixels in original_tdTomato")
ax[2].set_ylabel("value of pixels in network_generated_AFtd")
cb = fig.colorbar(im, ax=ax)
cb.set_label('$log_{10}(pixels)$')

plt.show()


# In[15]:
sg = 1.0
bac_td0 = gaussian(bac_td0, sigma=sg)
ratio0 = bac_td0/gfp
ratio0[ratio0 > 0.6] = 0
ratio0[gfp < 1.8 * autoFlu] = 0
ratio0[0][0] = 0.6
bac_td = gaussian(bac_td, sigma=sg)
ratio = bac_td/gfp
ratio[ratio > 0.6] = 0
ratio[gfp < 1.8 * autoFlu] = 0

fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharex=True, sharey=True)
ax = axes.ravel()

cmap = "turbo"

im = ax[0].imshow(ratio0, cmap=cmap)
ax[0].set_title("Theoretical relative tdTomato expression")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[1].imshow(ratio, cmap=cmap)
ax[1].set_title("CG relative tdTomato expression")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

ra = ratio0-ratio
ra[0][0] = -0.5
ra[0][1] = 0.5
im = ax[2].imshow(ra, cmap="RdBu")
ax[2].set_title("bias of relative tdTomato expression")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

for a in ax.ravel():
    a.axis('off')
fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharex=True, sharey=True)
ax = axes.ravel()

cmap = "turbo"

im = ax[0].imshow(bac_td0, cmap=cmap)
ax[0].set_title("Theoretical absolute tdTomato expression")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

im = ax[1].imshow(bac_td, cmap=cmap)
ax[1].set_title("CG absolute tdTomato expression")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

ra = bac_td0 - bac_td
#ra[gfp < 1.8 * autoFlu] = 0
ra[0][0] = -0.003
ra[0][1] = 0.003
im = ax[2].imshow(ra, cmap="RdBu")
ax[2].set_title("bias of absolute tdTomato expression")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="3%")
plt.colorbar(im, cax=cax)

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
plt.show()

lm0 = []
for j in np.linspace(0,0.5,200):
    temp = [i for i in ratio0[ratio0>j]]
    temp = np.array(temp)
    value = len([i for i in temp[temp<j+0.0025]])
    lm0.append(value)
temp = [i for i in ratio0[ratio0>0]]
temp = np.array(temp)

lm = []
for j in np.linspace(0,0.5,200):
    temp = [i for i in ratio[ratio>j]]
    temp = np.array(temp)
    value = len([i for i in temp[temp<j+0.0025]])
    lm.append(value)
temp = [i for i in ratio[ratio>0]]
temp = np.array(temp)

plt.scatter(np.linspace(0,0.5,200), lm, c="g", s=6, alpha=0.5)
#plt.scatter(np.linspace(0,0.5,200), lm0, c="gray", s=6, alpha=0.5)
plt.xlabel("fluorescent strength per pixel")
plt.ylabel("amount of pixels") # range at 0~0.5 splited into 200 parts for counting
plt.show()
