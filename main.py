#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:49:34 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

Credits are given to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from data_import import data_pooler, NDStandardScaler
# !!! Import here the model you need, follow: from <model_name> import Generator, Discriminator, train_model
from wgan_gp import Generator, Discriminator, train_model

cuda = True if torch.cuda.is_available() else False
print('cuda: ', cuda)
torch.manual_seed(0)

#%% Initializations of variables

# Data parameters
latent_dim = 100
height = 16
width = 76
depth = 1
image_shape = (depth, height, width)

# Training parameters
batch_size = 32
lr = 0.00005
num_epochs= 1000
lambda_gp = 10
n_discriminator = 5
saving_interval = 50

#%% Data import


ss1 = data_pooler(dataset_name = 'TenHealthyData', augment = False)

for ii in range(len(ss1)):
    if ii == 0:
        x_train = np.array(ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 1])
    else:
        x_train = np.concatenate((x_train, ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 1]))

del ss1

# =============================================================================
# Standard scaling
# =============================================================================
scaler = NDStandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

# x_train = np.expand_dims(x_train, axis=1)[:, :, :76, :]        
x_train = torch.unsqueeze(torch.from_numpy(x_train), axis=1)[:,:,:,:76]
y_train = torch.ones((x_train.shape[0],1))

train_dat = TensorDataset(x_train, y_train.type(dtype = torch.long))

train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)
    
#%% Model initialization
generator = Generator(latent_dim, image_shape)
discriminator = Discriminator(latent_dim, image_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()

optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=lr)
optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#%% Model Training

train_model(train_loader, generator, discriminator, optimizer_generator, 
            optimizer_discriminator, num_epochs, latent_dim, lambda_gp, n_discriminator, 
            Tensor, batch_size, saving_interval)

