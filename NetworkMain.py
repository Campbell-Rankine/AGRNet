#!/usr/bin/env python
# coding: utf-8

# # Construct GAN and structure
# ---
# Construct GAN, define training loop with multithreaded approach, and practice using industry standard terminal initialization commands

# In[161]:


### - IMPORTS - ###
from PIL import Image
import numpy as np
import glob
from multiprocessing.dummy import Pool as TP
import cv2 as cv
import os
import random
### - other data augmentation imports - ### (if needed)
### - Imports - ###
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim #Neural network imports, multiply data etc
from torchvision.transforms import ToTensor
import torchvision.models as models
import torchvision
import torch.nn.functional as F #Neural Network used in Comp4660 at ANU

from Utils.NetworkHelpers import EqualizedLR_Conv2d, Pixel_norm, Minibatch_std

### - Other global variables - ###
LOVTV = [15, 26, 66] ##Training values to leave out

img_folder = '/Users/campb/Documents/PersonalProjects/AGRNet/Dataset/'

NS = '/Sample-'

image_format = 'RGB'

imsize = 4

multiplication_factor = 20

num_channels=3
kernal=4
s=2
p=1


# In[171]:


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
T.manual_seed(manualSeed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class fRGB(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cvt = EqualizedLR_Conv2d(in_c, out_c, (1,1), stride=(1,1))
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.cvt(x)
        return self.relu(x)
        
class tRGB(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.cvt = EqualizedLR_Conv2d(in_c, out_c, (1,1), stride=(1,1))
        
    def forward(self, x):
        return(self.cvt(x))


# In[173]:


#Discriminator block
class D_Cell(nn.Module):
    def __init__(self, in_c, out_c, sb=0):
        self.sb = sb
        super().__init__()
        
        #Define network structure                                                         #initial block b (1-alpha)
        if sb == 0:
            #Set normal cell structure
            self.econv1 = EqualizedLR_Conv2d(in_c, out_c, (3,3), stride=(1,1), padding=(1,1)) #Initial block a (alpha)
            
            self.econv2 = EqualizedLR_Conv2d(out_c, out_c, (3,3), stride=(1,1), padding=(1,1))
            
            self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            self.mbstd = Minibatch_std()
            self.econv1 = EqualizedLR_Conv2d(in_c+1, out_c, (3,3), stride=(1,1), padding=(1,1)) #Initial block a (alpha)
            self.econv2 = EqualizedLR_Conv2d(out_c, out_c, (3,3), stride=(1,1), padding = (1,1)) #output block
            self.flat = nn.Flatten()
            self.lin = nn.Linear(16*out_c, 1) #We multiply by 16 since our first image will always be 4x4, therefore this flatten will always lead to this value
        self.relu = nn.LeakyReLU(0.2)
                #Weight inititalization
        if sb == 0:
            nn.init.normal_(self.econv1.weight)
            nn.init.zeros_(self.econv1.bias)
        nn.init.normal_(self.econv2.bias)
        nn.init.zeros_(self.econv2.bias)
    # ,nn.LeakyReLU(0.2, inplace=True), nn.Linear(out_c, 1)
    def forward(self, x):
        ### - Account for each discriminator block archetype - ###
        if self.sb == 0:
            x = self.econv1(x)
            x = self.relu(x)
            
            x = self.econv2(x)
            x = self.relu(x)
            
            x = self.outlayer(x)
        else:
            x = self.mbstd(x)
            x = self.econv1(x)
            x = self.relu(x)
            x = self.econv2(x)
            x = self.relu(x)
            x = self.flat(x)
            x=self.lin(x)
        return x
            
#Generator Block

class G_Cell(nn.Module):
    def __init__(self, in_c, out_c, sb=0):
        self.sb = sb
        super().__init__()
        
        #Define network structure
        if sb == 0:
            self.us = nn.Upsample(scale_factor=2, mode='nearest') #Base block (standard cell)
            self.conv1 = EqualizedLR_Conv2d(in_c, out_c, (3,3), stride=(1,1), padding=(1,1))
        else:
            self.conv1 = EqualizedLR_Conv2d(in_c, out_c, (4,4), stride=(1,1), padding=(3,3))
        self.conv2 = EqualizedLR_Conv2d(out_c, out_c, (3,3), stride=(1,1), padding=(1,1))

        
        self.relu = nn.LeakyReLU()
        self.pn = Pixel_norm()
        #Weight inititalization
        nn.init.normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        if sb == 0:
            nn.init.normal_(self.conv2.bias)
            nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        if self.sb == 0:
            x = self.us(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pn(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pn(x)
        return x
            


# # Network Structure and Basic Theory
# 
# ---
# 
# Each network will progressively need to grow more and more in order to upscale the images while keeping each of the dims the same for upscaling it.
# Therefore to accurately train this model we construct it in such a way that we may output a 1 megapixel image.
# 
# This involves defining the structure for the overall network once it is finished and including a depth index variable that will be increased in training.

# In[174]:


#Discriminator and Generator#
class G(nn.Module):
    def __init__(self, ls, out):
        """
        ls is latent size
        out is desired output resolution
        build structure iteratively
        """
        super().__init__()
        self.depth = 1 #Current indexing
        self.alpha = 0 #Fade value
        self.incalpha = 0 #Value to increment alpha by
        
        self.trgb = tRGB(ls, 3) #torgb value
        self.us = nn.Upsample(scale_factor=2, mode='nearest')
        self.net = nn.ModuleList([G_Cell(ls, ls, sb=1)])
        self.rgbs = nn.ModuleList([tRGB(ls, 3)])
        
        #Add all standard blocks
        for i in range(2, int(np.log2(out))):
            ### - trick is to decrease the latent vector as well for each of the higher level blocks - ###
            in_c = ls
            out_c = ls
            self.net.append(G_Cell(in_c, out_c))
            self.rgbs.append(tRGB(out_c, 3))
            
    def forward(self, x):
        for cell in self.net[:self.depth-1]:
            x = cell(x)
        out = self.net[self.depth-1](x)
        crgb = self.rgbs[self.depth-1](out)
        if self.alpha > 0:
            xprev = self.us(x)
            rgbprev = self.rgbs[self.depth-2](xprev)
            crgb = (1-self.alpha) * (rgbprev) + (self.alpha)*(crgb)
            self.alpha += self.incalpha
        return crgb
    def inc_depth(self, iters):
        self.incalpha = 1/iters
        self.alpha = 1/iters
        self.depth += 1

class D(nn.Module):
    def __init__(self,ls, out):
        super().__init__()
        self.depth = 1
        self.alpha = 0
        self.incalpha = 0
        
        self.relu = nn.LeakyReLU(0.2)
        self.ds = nn.AvgPool2d(2, stride=(2,2))
        
        self.net = nn.ModuleList([D_Cell(ls, ls, sb=3)]) #initialize final block
        self.frgbs = nn.ModuleList([fRGB(3, ls)])
        
        for i in range(2, int(np.log2(out))):
            in_c, out_c = ls, ls
                
            self.net.append(D_Cell(in_c, out_c))
            self.frgbs.append(fRGB(3, in_c))
            
    def forward(self, x):
        xc = self.frgbs[self.depth-1](x)
        xc = self.net[self.depth-1](xc)
        if self.alpha > 0: #if depth != 1
            x = self.ds(x)
            xprev = self.frgbs[self.depth-2](x)
            xprev = self.relu(xprev)
            #xprev = self.ds(xprev)
            xc = (1-self.alpha)*xprev + (self.alpha)*xc
            self.alpha += self.incalpha
        for cell in reversed(self.net[:self.depth-1]):
            xc = cell(xc)

        return xc

    def inc_depth(self, iters):
        self.incalpha = 1/iters
        self.alpha = 1/iters
        self.depth += 1
        


# In[ ]:




