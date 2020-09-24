#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:49:34 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

Credits are given to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

"""

# import os
import numpy as np
# import matplotlib.pyplot as plt

# from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
import torch
torch.cuda.empty_cache()

from data_import import data_pooler, NDStandardScaler
# Comment out the needed model and comment the rest of import lines

cuda = True if torch.cuda.is_available() else False
# if cuda has runtime error, then run "nvidia-modprobe -u" in terminal (first download nvidia-modprobe in ubuntu)
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
gan_type = 'wgan_gp'
if gan_type == 'wgan_gp':
    from wgan_gp import Generator, Discriminator, train_model
elif gan_type == 'dcgan':
    from dcgan import Generator, Discriminator, train_model, weights_init_normal

batch_size = 32
lr = 0.00005
num_epochs= 50
lambda_gp = 10
n_discriminator = 5
saving_interval = 5

#%%

def gan_train_generate(sub_idxs = [0,1,2,3,4,5,6,7,8,9], 
                       no_samples_to_generate = 1152,
                       gan_type = 'dcgan', 
                       signal_type_to_generate = 'Target'):
    
    ss1 = data_pooler(dataset_name = 'TenHealthyData', augment = False)
    target = 1 if signal_type_to_generate == 'Target' else 0
    
    if len(sub_idxs) == 1:
        x_train = np.array(ss1[sub_idxs[0]][0]['xtrain'][ss1[sub_idxs[0]][0]['ytrain'] == target][:288])
    else:
        for ii in sub_idxs:
            if ii == 0:
                x_train = np.array(ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == target][:288])
            else:
                x_train = np.concatenate((x_train, ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == target][:288]))
    del ss1
    
    # =============================================================================
    # Standard scaling
    # =============================================================================
    # scaler = NDStandardScaler()
    # x_train = scaler.fit_transform(x_train)
    
    # =============================================================================
    # Min-max normalization
    # =============================================================================
    x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))    

    # Moving data to torch
    # x_train = np.expand_dims(x_train, axis=1)[:, :, :76, :]        
    x_train = torch.unsqueeze(torch.from_numpy(x_train), axis=1)[:,:,:,:76]
    y_train = torch.ones((x_train.shape[0],1))
    
    train_dat = TensorDataset(x_train, y_train.type(dtype = torch.long))
    
    train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)
        
    # =============================================================================
    #     Model initialization
    # =============================================================================
    
    generator = Generator(latent_dim, image_shape)
    discriminator = Discriminator(latent_dim, image_shape)
    
    # Depending on type of GAN, make training:
    if gan_type == 'dcgan':
        adversarial_loss = torch.nn.BCELoss()
        
        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
        
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # =============================================================================
        #     Model Training
        # =============================================================================
        
        generator, discriminator = train_model(train_loader, generator, discriminator, optimizer_generator, 
                    optimizer_discriminator, adversarial_loss, num_epochs, latent_dim, 
                    Tensor, batch_size, saving_interval)     
        
        
    elif gan_type == 'wgan_gp':
        
        if cuda:
            generator.cuda()
            discriminator.cuda()
        
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.5))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.5))
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # =============================================================================
        #     Model Training
        # =============================================================================
        
        generator, discriminator = train_model(train_loader, generator, discriminator, optimizer_generator, 
                    optimizer_discriminator, num_epochs, latent_dim, lambda_gp, n_discriminator, 
                    Tensor, batch_size, saving_interval)
    
    # =============================================================================
    #     Generate samples from trained model
    # =====================================================zero_grad========================
    generated_erp = np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2]))
    del train_loader, train_dat
    
    for ii in range(no_samples_to_generate//batch_size):
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        generated_erp[batch_size*ii:batch_size*ii+batch_size, :, :, :] = generator(z).cpu().data.numpy()
        
    del y_train, generator, discriminator, Tensor
    
    return np.squeeze(x_train.cpu().data.numpy(), axis=1) , np.squeeze(generated_erp, axis=1)

#%%

subs = [0,1,3,4,5,6,7,8,9]
# for sub_idx in subs:
target_real, target_generated = gan_train_generate(subs, 1152, gan_type, 'Target')
nontarget_real, nontarget_generated = gan_train_generate(subs, 1152, gan_type, 'NonTarget')


#%% Let's do some quality tests

# =============================================================================
# GAN-test (accuracy test: train on generated samples, test on real samples)
# =============================================================================
from gan_test import gan_test

real_combined = {'x': np.concatenate((target_real, nontarget_real)), 
                     'y': np.concatenate((np.ones(target_real.shape[0],), np.zeros(nontarget_real.shape[0],)))}

generated_combined = {'x': np.concatenate((target_generated, nontarget_generated)), 
                     'y': np.concatenate((np.ones(target_generated.shape[0],), np.zeros(nontarget_generated.shape[0],)))}

accuracy_LDA = gan_test(real_combined, generated_combined, 'LDA')
accuracy_LR = gan_test(real_combined, generated_combined, 'LogisticRegression')
accuracy_SVM = gan_test(real_combined, generated_combined, 'SVM')
accuracy_CNN = gan_test(real_combined, generated_combined, 'DCNN')

print(accuracy_LDA, accuracy_LR, accuracy_SVM, accuracy_CNN)


# KL-divergence test


#%% Visualization t-SNE test
from gan_test import t_sne, t_sne_one_data

sns_plot = t_sne(real_combined, generated_combined)

# sns_plot = t_sne_one_data(real_combined)
# sns_plot = t_sne_one_data(generated_combined)








