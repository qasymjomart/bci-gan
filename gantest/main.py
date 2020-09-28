#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:49:34 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

Credits are given to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

"""

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import torch
torch.cuda.empty_cache()

from data_import import Data_loader
from gan_test import gan_test, t_sne


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
gan_type = 'dcgan'
if gan_type == 'wgan_gp':
    from wgan_gp import Generator, Discriminator, train_model
elif gan_type == 'dcgan':
    from dcgan import Generator, Discriminator, train_model, weights_init_normal

batch_size = 32
lr = 0.0001
num_epochs= 50
lambda_gp = 10
n_discriminator = 5
saving_interval = num_epochs/10

#%%

subs = [0,1,2,3,4,5,6,7,8,9]
real_data, generated_data = {}, {}

for target in ['Target', 'NonTarget']:
    
    data_load = Data_loader(dataset_name = 'TenHealthyData')
    x_train = data_load.pool_one_class(target, normalize=True)
    
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
        
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0, 0.9))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0, 0.9))
        
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
    no_samples_to_generate = len(x_train)
    generated_erp = np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2]))
    del train_loader, train_dat
    
    for ii in range(no_samples_to_generate//batch_size):
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        generated_erp[batch_size*ii:batch_size*ii+batch_size, :, :, :] = generator(z).cpu().data.numpy()
        
    del y_train, generator, discriminator, Tensor
    
    real_data[target] = np.squeeze(x_train.cpu().data.numpy(), axis=1)
    generated_data[target] = np.squeeze(generated_erp, axis=1)
    del x_train, generated_erp
    torch.cuda.empty_cache()


#%% Let's do some quality tests

# =============================================================================
# GAN-test (accuracy test: train on generated samples, test on real samples)
# =============================================================================

real_combined = {'x': np.concatenate((real_data['Target'], real_data['NonTarget'])), 
                     'y': np.concatenate((np.ones(real_data['Target'].shape[0],), np.zeros(real_data['NonTarget'].shape[0],)))}

generated_combined = {'x': np.concatenate((generated_data['Target'], generated_data['NonTarget'])), 
                     'y': np.concatenate((np.ones(generated_data['Target'].shape[0],), np.zeros(generated_data['NonTarget'].shape[0],)))}

accuracy_LDA = gan_test(real_combined, generated_combined, 'LDA')
accuracy_LR = gan_test(real_combined, generated_combined, 'LogisticRegression')
accuracy_SVM = gan_test(real_combined, generated_combined, 'SVM')
accuracy_CNN = gan_test(real_combined, generated_combined, 'DCNN')

print(accuracy_LDA, accuracy_LR, accuracy_SVM, accuracy_CNN)
with open('gan_test_results.txt', 'w') as f:
    f.write('GAN type: ' + gan_type)
    f.write('accuracy on LDA: ' + str(accuracy_LDA))
    f.write('accuracy on LR: ' + str(accuracy_LR))
    f.write('accuracy on SVM ' + str(accuracy_SVM))
    f.write('accuracy on CNN ' + str(accuracy_CNN))


# Visualization t-SNE test

sns_plot = t_sne(real_combined, generated_combined)

# sns_plot = t_sne_one_data(real_combined)
# sns_plot = t_sne_one_data(generated_combined)









