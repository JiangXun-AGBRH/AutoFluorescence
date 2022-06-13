#!/usr/bin/env python
# coding: utf-8
import torch # a deeplearning package developed by Facebook
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
from skimage.filters import gaussian 
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # tools to make zoomin or zoomout possible in the heatmap
import matplotlib
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dtype = torch.float32
if torch.cuda.is_available():
    print("in this study, cuda will be used.\n")
    device = torch.device("cuda:0") # if cuda is available, use cuda cores for CNN training. cuda cores can greatly increase the training speed
else:
    print("cuda is not available in this environment, cpu will be used.\n")
    device = torch.device("cpu") # if cuda is not available, use the cpu for training. All computers have cpu/cpus. 

def shiftdown(arr, n): # n lines will be shifted
    w, h = arr.shape
    addtion = np.zeros((n, w))
    new_arr = np.concatenate((addtion, arr[:-n, :]), axis=0)
    return new_arr

def shiftup(arr, n): # n lines will be shifted
    w, h = arr.shape
    addtion = np.zeros((n, w))
    new_arr = np.concatenate((arr[n:, :], addtion), axis=0)
    return new_arr

def shiftleft(arr, n): # n lines will be shifted
    w, h = arr.shape
    addtion = np.zeros((h, n))
    new_arr = np.concatenate((arr[:, n:], addtion), axis=1)
    return new_arr

def shiftright(arr, n): # n lines will be shifted
    w, h = arr.shape
    addtion = np.zeros((h, n))
    new_arr = np.concatenate((addtion, arr[:, :-n]), axis=1)
    return new_arr

def checkInput(path):
    folders = listdir(path)
    if len(folders) == 3:
        folder1, folder2, folder3 = [], [], []
        L = [folder1, folder2, folder3]
    elif len(folders) == 2:
        folder1, folder2 = [], []
        L = [folder1, folder2]
    try:
        n = 0
        for folder in L:
            folder = [file.split("-")[1] for file in listdir(path+"/"+folders[n])]
            #print(f'you {path} data is here: {path}\\{folders[n]}\\{folder[:]}')
            n += 1
    except:
        folder1 = [0]
        print("ERROR; make sure you put all the needed data in the \"%s\" folder!" % path)
    
    if len(folders) == 2:
        folder3 = folder1

    if (folder1 == folder2) and (folder1 == folder3):
        print("all data in the \"%s\" folder seem proper." % path)

        
def checkTraindata(path):
    for i in listdir(path):
        new_path = path + "/" + i
        checkInput(new_path)
        print()
        
def imageShaker(im, S, Q):
    if S == 0:
        fig = shiftdown(im, Q)
    elif S == 1:
        fig = shiftleft(im, Q)
    elif S == 2:
        fig = shiftright(im, Q)
    elif S == 3:
        fig = shiftup(im, Q)
    else:
        fig = im
    return fig


checkInput("input")
print()
checkTraindata("training_dataset")
input("Please read the README file before starting the process. if you already read the README please press Enter to continue.\n")
sg = input("please input here the sigma factor for the gaussian filter (0.6~1.2 is suggested, higher value will make the image more smooth): ", )
print()
sg = float(sg)
print("gaussian sigma factor in this study is:", sg)
print()

def training_im_generator(sg):
    angles = [0,90,180,270]
    # all the images files in different channels
    root_td = sorted(listdir("training_dataset/plant_tissue/tdTomato/"))
    root_gfp = sorted(listdir("training_dataset/plant_tissue/GFP/"))
    root_af = sorted(listdir("training_dataset/plant_tissue/Autofluorescence/"))
    bac_td = sorted(listdir("training_dataset/bacteria/tdTomato/"))
    bac_gfp = sorted(listdir("training_dataset/bacteria/GFP/"))
    for i in range(len(root_af)):
        for j in range(len(bac_td)):
            #read all the images files in different channels as nparray
            td_root = io.imread("training_dataset/plant_tissue/tdTomato/"+root_td[i])
            gfp_root = io.imread("training_dataset/plant_tissue/GFP/"+root_gfp[i])
            af_root = io.imread("training_dataset/plant_tissue/Autofluorescence/"+root_af[i])
            td_bac0 = io.imread("training_dataset/bacteria/tdTomato/"+bac_td[j])
            gfp_bac0 = io.imread("training_dataset/bacteria/GFP/"+bac_gfp[j])

            number = random.randint(0,3)
            R = angles[number]
            number = random.randint(0,3)
            K = angles[number]
            N = random.randint(0,1)
            M = random.randint(0,1)

            if M == 0:
                td_bac0 = np.flip(td_bac0, axis=N)
                gfp_bac0 = np.flip(gfp_bac0, axis=N)
                
            N = random.randint(0,1)
            M = random.randint(0,1)
            if M == 0:
                td_root = np.flip(td_root, axis=N)
                gfp_root = np.flip(gfp_root, axis=N)
                af_root = np.flip(af_root, axis=N)

            td_bac0 = rotate(td_bac0, K)
            gfp_bac0 = rotate(gfp_bac0, K)
            td_root = rotate(td_root, R)
            gfp_root = rotate(gfp_root, R)
            af_root = rotate(af_root, R)
            

            gfp_root = gaussian(gfp_root, sigma=sg)
            af_root = gaussian(af_root, sigma=sg)
            gfp_bac0 = gaussian(gfp_bac0,sigma=sg)
            td_root = gaussian(td_root, sigma=sg)
            td_bac0 = gaussian(td_bac0, sigma=sg)
            #td_root[td_root<1e-4] = 1e-4
            #td_bac0[td_bac0<1e-4] = 1e-4
            S = random.randint(0, 3)
            Q = random.randint(1, 150)
            gfp_bac = imageShaker(gfp_bac0.copy(), S, Q)
            td_bac = imageShaker(td_bac0.copy(), S, Q)
            S = random.randint(0, 3)
            Q = random.randint(1, 150)
            gfp_root = imageShaker(gfp_root.copy(), S, Q)
            af_root = imageShaker(af_root.copy(), S, Q)
            td_root = imageShaker(td_root.copy(), S, Q)
            td_root[td_root<1e-5] = 1e-5

            tempBcTd = td_bac.copy() * random.uniform(0.8, 4)
            tdTomato = td_root.copy() + tempBcTd
            tdTomato[tdTomato>1] = 1
            tdTomato[tdTomato<1e-5] = 1e-5
            
            tempBcGFP = gfp_bac.copy() * random.uniform(3, 5)
            gfp = gfp_root.copy() + tempBcGFP
            gfp[gfp>1] = 1
            #gfp = gaussian(gfp, sigma=sg)
            
            bacFilter = np.ones_like(gfp_bac)
            bacFilter[gfp_bac>gfp_bac.mean()] = 0
            edge_sobel = filters.sobel(bacFilter)
            edge = edge_sobel.copy()
            edge[edge>0] = 1.5*gfp_bac.mean()
            edge_bacter = edge + gfp_bac
            bF = (edge_bacter > gfp_bac.mean())
            bacFilter = np.ones_like(gfp_bac)
            bacFilter[bF] = 0
            
            td_filtered = tdTomato.copy()
            td_filtered[bacFilter==0] = 0
            autoFlu = af_root.copy()
            autoFlu[td_filtered==1] = 0
            factor = td_root.mean()/af_root.mean()
            td_new = td_filtered + autoFlu * factor
            td_new[td_new>1] = 1

            td_root = np.array(td_root, dtype=np.float32)
            td_root = td_root.reshape((1,1,512,512))
            td_root = torch.tensor(td_root, dtype=dtype, device=device)
            
            image = np.concatenate((td_filtered, gfp, af_root), dtype=np.float32)
            image = image.reshape((3,512,512))
            image = torch.tensor(image, dtype=dtype, device=device)
            image = image.unsqueeze(0)
            
            yield image, td_root, bacFilter, tempBcTd, tempBcGFP

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

def readInput(sg):
    td = io.imread("input/tdTomato/" + listdir("input/tdTomato/")[0])
    gfp = io.imread("input/GFP/" + listdir("input/GFP/")[0])
    af = io.imread("input/Autofluorescence/" + listdir("input/Autofluorescence/")[0])
    
    gfp = gaussian(gfp, sigma=sg)
    td = gaussian(td, sigma=sg)
    af = gaussian(af, sigma=sg)
    #gfp[gfp<1e-4] = 1e-4
    #af[af<1e-4] = 1e-4
    td[td<1e-5] = 1e-5
    return td, gfp, af

def GetFilter(sg, factor):
    td, gfp, af = readInput(sg)
    autoFlu = gfp.mean()
    bacteria_filter = gfp > autoFlu * factor
    bcfilter = np.ones_like(gfp)
    bcfilter[bacteria_filter] = 0

    edge_sobel = filters.sobel(bcfilter)
    edge = edge_sobel.copy()
    edge[edge>0] = 1.5*factor*autoFlu
    edge_bacter = edge + gfp
    bF = (edge_bacter > factor*autoFlu)
    bcfilter = np.zeros_like(gfp)
    bcfilter[bF] = 1 

    edge_sobel = filters.sobel(bcfilter)
    edge = edge_sobel.copy()
    edge[edge>0] = 1.5*factor*autoFlu
    edge_bacter = edge + gfp
    bF = (edge_bacter > factor*autoFlu)
    bcfilter = np.ones_like(gfp)
    bcfilter[bF] = 0 

    return bcfilter

print("\nthe smaller the factor (bacteria-filter threshold) is, the more signals will be filtered/removed")
print("\nSoon a popup window will show you some images, close the window it the program will continue.")
factor = input("\nplease give a threshold for the bacteria-filter (can start with 1.5, change it according to the performance of the filter):", )

factor = float(factor)

ok = str(factor)
while ok.upper() != "OK":
    td, gfp, af = readInput(sg)

    autoFlu = gfp.mean()
    bacteria_filter = gfp > autoFlu * factor
    bcfilter = np.ones_like(gfp)
    bcfilter[bacteria_filter] = 0

    edge_sobel = filters.sobel(bcfilter)
    edge = edge_sobel.copy()
    edge[edge>0] = 1.5*factor*autoFlu
    edge_bacter = edge + gfp
    bF = (edge_bacter > factor*autoFlu)
    bcfilter = np.zeros_like(gfp)
    bcfilter[bF] = 1 

    edge_sobel = filters.sobel(bcfilter)
    edge = edge_sobel.copy()
    edge[edge>0] = 1.5*factor*autoFlu
    edge_bacter = edge + gfp
    bF = (edge_bacter > factor*autoFlu)
    bcfilter = np.ones_like(gfp)
    bcfilter[bF] = 0 
    # check the bacteria-depletion-filter, if the filter is not good, change the factor. 
    fig, axes = plt.subplots(1,3, figsize=(18,6), facecolor='w', edgecolor='k', sharex=True, sharey=True, dpi=90)
    ax = axes.ravel()

    fig.suptitle("Close this window to continue")
    im = ax[0].imshow(gfp, cmap='binary')
    ax[0].set_title("original GFP")
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="3%", pad=0)
    plt.colorbar(im, cax=cax)
    
    im = ax[1].imshow(bcfilter, cmap='binary')
    ax[1].set_title("bacteria-filter")
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="3%", pad=0)
    plt.colorbar(im, cax=cax)    
    
    temp = bcfilter * td
    im = ax[2].imshow(temp, cmap='binary')
    ax[2].set_title("tdTomato without bacteria")
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="3%", pad=0)
    plt.colorbar(im, cax=cax)
    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    plt.show()
    print("\nthe smaller the factor (bacteria-filter threshold) is, the more signals will be filtered/removed")
    ok = input("\nis the bacteria-filter sufficient to remove all the bacteria signals? If it is, please write OK \nif not please write the new factor. OK or a new value: ", )
    try:
        factor = float(ok)
    except:
        pass
print(f'the used factor for bacteria-filter is {factor}')


# hyperparameters could be changed. In this study learning_rate was suggest to be at 1e-3 to 1e-5. If the MSELoss could not be reduced but still greater than 0.1, 1e-4 or 1e-5 could be tried. 
# if you want to use a pre-trained Model for further training, please set the learning rate to 1e-4 or 1e-5.
loss_fn = torch.nn.MSELoss(reduction='sum') #here is the sum of MSELoss
learning_rate = 2e-5 # if it is trained from scratch, use 1e-3 to save time. 
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

checkpoint = input("\nthe learning rate was set to %.5f as default in the script, do you want to change it? Yes or No:" %(learning_rate))
if checkpoint.lower() == "yes":
    learning_rate = float(input("\nplease write the new learning rate here:", ))
else:
    learning_rate = learning_rate

epoch = int(input("\nhow many cycles do you want to train the model (suggested 100-2000 cycles):"))

pretrained = "cnn_forPaper.plk"
if torch.cuda.is_available():
    net.load_state_dict(torch.load(pretrained))
else:
    net.load_state_dict(torch.load(pretrained, map_location=device))

loss_list = []
plt.ion()
for t in range(epoch): # It is recommented to save the parameters each 50~200 cycles. If the output is not good enough, reload the saved Model and start training again, and save the Model in another file. Do this until the MSELoss is smaller than 0.05 or even lower.  
    for image, label, bacFilter, tempBcTd, tempBcGFP in training_im_generator(sg):
        # training with Strategy II
        number = random.randint(1,100)
        # training with Strategy I
        if number%1 == 0: # control the ratio between Generator I and II training
            y = net(image) 
            loss = loss_fn(y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        number = random.randint(1,100)
        # training with Strategy I
        if number%1 == 0: # control the ratio between Generator I and II training
            n = random.randint(0,5)
            L = [0,0,0,90,180,270]
            td, gfp, af = readInput(sg)
            td = rotate(td, L[n])
            gfp = rotate(gfp, L[n])
            af = rotate(af, L[n])
            bf = rotate(bcfilter, L[n])
            tdTomato = td.copy() * bacFilter 
            tdTomato[bf==0] = 0
            tdPlusBc = td.copy() + tempBcTd
            tdPlusBc[tdPlusBc>1] = 1
            #tdPlusBc = gaussian(tdPlusBc, sigma=1)
            #tdPlusBc[tdPlusBc<1e-4] = 1e-4
            gfpPlusBc = gfp.copy() + tempBcGFP
            gfpPlusBc[gfpPlusBc>1] = 1
            #gfpPlusBc = gaussian(gfpPlusBc, sigma=1)
            #gfpPlusBc[gfpPlusBc<1e-4] = 1e-4
            #af = gaussian(af, sigma=1)
            #td = gaussian(td, sigma=1)

            bac_filter = torch.tensor(bf, device=device, dtype=dtype)
            bac_filter = bac_filter.reshape((1,512,512))
            bac_filter = bac_filter.unsqueeze(0)

            fig = np.concatenate((tdTomato, gfpPlusBc, af), dtype=np.float32)
            fig = torch.tensor(fig, device=device, dtype=dtype)
            fig = fig.reshape((1,3,512,512))

            
            td = td * bf
            td = torch.tensor(np.array(td, dtype=np.float32), device=device, dtype=dtype)
            td = td.reshape((1,512,512))
            td = td.unsqueeze(1)
            #td = td * bac_filter

            y = net(fig)
            y = y * bac_filter # remove the pixels which have overlap with the bacteria
            loss = loss_fn(y, td)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    loss_list.append([loss.item()])
    j = loss_list
    td, gfp, af = readInput(sg)
    tdTomato = td.copy()
    tdTomato[bcfilter==0] = 0

    fig = np.concatenate((tdTomato, gfp, af), dtype=np.float32)
    fig = torch.tensor(fig, device=device, dtype=dtype)
    fig = fig.reshape((1,3,512,512))

    if torch.cuda.is_available():
        AF = net(fig).cpu().data.numpy() # predicted AF and transform it in to numpy array.
    else:
        AF = net(fig).data.numpy() # predicted AF and transform it in to numpy array.
    AF = AF.reshape((512,512))

    td, gfp, af = readInput(sg)
    plt.subplot(2,1,2)
    plt.plot(j)
    #plt.draw()
    plt.title("MSELoss")
    plt.subplot(2,2,1)
    plt.title("Output/Predict")
    plt.imshow(AF, cmap="turbo")
    plt.subplot(2,2,2)
    plt.title("tdTomato Input")
    plt.imshow(tdTomato, cmap="turbo")
    plt.draw()
    plt.pause(0.0002)
    plt.clf()

    if t % 10 == 0:
        print(f"MSELoss at epoch {t} is:\t {loss.item()}")
        print("In progress now, please be patient and don't disturb!\n")
        
    if loss.item() < 0.001:
        torch.save(net.state_dict(),"cnn_forPaper_%d.plk" %(t+1))
        print("all the parameters of your trained model is save in file: cnn_forPaper_%d.plk" %(t+1))
        break
plt.close()

plt.plot(j)
plt.title("MSELoss")
plt.show()
torch.save(net.state_dict(),"cnn_forPaper_%d.plk" %(t+1))
print("all the parameters of your trained model is save in file: cnn_forPaper_%d.plk" %(t+1))

plk= "cnn_forPaper_%d.plk" %(t+1)
net.load_state_dict(torch.load(plk))


td, gfp, af = readInput(sg)
tdTomato = td.copy()
tdTomato[bcfilter==0] = 0

image = np.concatenate((tdTomato, gfp, af), dtype=np.float32) # concatenate three channels into one numpy array
image = image.reshape((3,512,512)) # reshape the input to 4*512*512
image = torch.tensor(image, dtype=dtype, device=device) # put the array into device, here is cpu. The data is transformed into Tensor. 
image = image.unsqueeze(0) # unsqueeze the data, while the pytorch can only uptake data as minibatch

if torch.cuda.is_available():
    AF = net(image).cpu().data.numpy() # predicted AF and transform it in to numpy array.
else:
    AF = net(image).data.numpy() # predicted AF and transform it in to numpy array.
AF = AF.reshape((512,512)) # reshape the AF into a 512*512 image, or we could understant the output

bac_td = td - AF # baterical tdTomato should be the AF removed orignal tdTomato 

# make those image have the same colormap range. 
threshold = 0.4
checkpoint = input("the default max value for autofluoresence in the image was set to 0.4, if you want to change it please input a new value here:\nor please write ok.")
if checkpoint.lower() != "ok":
    threshold = float(checkpoint)
else:
    threshold = 0.4
af[af>threshold]=threshold #few pixels may be over-exposured. A max value should be invited in order to make all pixels in the proper range of a colormap.
# the range should be 0~threshold, set different channels at the same range to make it easier to compare them. 
AF[0][0]=threshold 
#AF[AF<1e-4] = 1e-4
AF[0][1]=0
td[0][0]=threshold
td[0][1]=0
#af[0][0]=threshold 
af[0][1]=0

print("I will generate the output for you, all the images will be saved later. for the showup figure, you can move, zoomin, crop and save it in a different folder.")
#########################################################################################
# show the predicted results in images. The new matplotlib package may give warning reports, just ignore them. 
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True, dpi=150)
ax = axes.ravel()

cmap = 'turbo'#'terrain'#'turbo' several other colormaps could also be used. 
cmap01 = "gray"
im = ax[0].imshow(AF, cmap=cmap)
matplotlib.image.imsave('output/computerGeneratedAutofluorescence.png', AF, cmap=cmap01)
np.save("output/computerGeneratedAutofluorescence.nparray", AF)
ax[0].set_title("network generated AAFtd")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%", pad=0)
plt.colorbar(im, cax=cax)

im = ax[1].imshow(td, cmap=cmap)
matplotlib.image.imsave('output/tdTomato.png', td, cmap=cmap01)
np.save("output/tdTomato.nparray", td)
ax[1].set_title("tdTomato channel")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%", pad=0)
plt.colorbar(im, cax=cax)

#gfp[0][0] = af.max()
im = ax[2].imshow(gfp, cmap=cmap)
matplotlib.image.imsave('output/GFP.png', gfp, cmap=cmap01)
np.save("output/GFP.nparray", gfp)
ax[2].set_title("GFP channel")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="3%", pad=0)
plt.colorbar(im, cax=cax)

im = ax[3].imshow(af, cmap=cmap)
matplotlib.image.imsave('output/autofluorescence.png', af, cmap=cmap01)
np.save("output/autofluorescence.nparray", af)
ax[3].set_title("AF channel")
divider = make_axes_locatable(ax[3])
cax = divider.append_axes("right", size="3%", pad=0)
plt.colorbar(im, cax=cax)

im = ax[4].imshow(bcfilter, cmap="binary")
matplotlib.image.imsave('output/bacteriaFilter.png', bcfilter, cmap=cmap01)
np.save("output/bacteriaFilter.nparray", bcfilter)
ax[4].set_title("bacterial filter")
divider = make_axes_locatable(ax[4])
cax = divider.append_axes("right", size="3%", pad=0)
plt.colorbar(im, cax=cax)

bac_td[bac_td<0] = 0 # set the value<0 to 0. after subtraction, some pixels would be smaller than 0, which doesn't any sense and should be set to 0. 

#bac_td[0][0] = af.max()
im = ax[5].imshow(bac_td, cmap=cmap)
matplotlib.image.imsave('output/AF_removed_image.png', bac_td, cmap=cmap01)
np.save("output/AF_removed_image.nparray", bac_td)
ax[5].set_title("tdTomato $-$ AAFtd $>$ 0")
divider = make_axes_locatable(ax[5])
cax = divider.append_axes("right", size="3%", pad=0)
plt.colorbar(im, cax=cax)

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
name = input("saving the image with name:", )
plt.savefig("output/"+name+".png", dpi=120)
plt.show()
