#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:09:27 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

AUGMENTATION: make Subject-specific augmentation and classification

"""

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch
torch.cuda.empty_cache()
from gan_test import t_sne

from data_import import Data_loader
from cnn import CNN, train_cnn_model
cuda = True if torch.cuda.is_available() else False
# if cuda has runtime error, then run "nvidia-modprobe -u" in terminal (first download nvidia-modprobe in ubuntu)
print('cuda: ', cuda)
torch.manual_seed(0)

#%%

# Data parameters
latent_dim = 100
height = 16
width = 76
depth = 1
image_shape = (depth, height, width)

# GAN Training parameters
gan_type = 'vae'
if gan_type == 'wgan_gp':
    from wgan_gp import Generator, Discriminator, train_model
elif gan_type == 'dcgan':
    from dcgan import Generator, Discriminator, train_model, weights_init_normal
elif gan_type == 'vae':
    from vae import VAE, train_model
batch_size = 32
lr = 0.0001
num_epochs= 1000
lambda_gp = 10
n_discriminator = 5
saving_interval = num_epochs/10
accuracies = []

no_samples_to_train_gan= 144
no_samples_to_generate = 144
#%% 

sub_idxs = [0,1,2,3,4,5,6,7,8,9]

data_load = Data_loader(dataset_name = 'TenHealthyData')
data, test_data = data_load.subject_specific(normalize = True)
#%%%

for sub in sub_idxs:
    
    generated_data = []
    for target in [0, 1]: # generate Target and Nontarget samples
        # Data preparation
        gan_train_data = np.array(data[sub]['xtrain'][data[sub]['ytrain'] == target])

        # Moving data to torch
        gan_train_data = torch.unsqueeze(torch.from_numpy(gan_train_data), axis=1)[:,:,:,:image_shape[2]]
        y_train = torch.ones((gan_train_data.shape[0],))
        
        train_dat = TensorDataset(gan_train_data, y_train.type(dtype = torch.long))
        train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)

        # =============================================================================
        #     GAN Model initialization
        # =============================================================================
        
        # Depending on type of GAN, make training:
        if gan_type == 'dcgan':
            generator = Generator(latent_dim, image_shape)
            discriminator = Discriminator(latent_dim, image_shape)
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
                        Tensor, batch_size, saving_interval, plotting=True)     
            
            
        elif gan_type == 'wgan_gp':
            generator = Generator(latent_dim, image_shape)
            discriminator = Discriminator(latent_dim, image_shape)
            
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

        elif gan_type == 'vae':
                        
                vae_model = VAE()
                vae_loss = torch.nn.MSELoss()
                
                if cuda:
                    vae_model.cuda()
                    vae_loss.cuda()
                    
                vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
                
                Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

                # =============================================================================
                #     Model Training
                # =============================================================================
                
                vae_model = train_model(train_loader, vae_model, 
                                        vae_optimizer, vae_loss, num_epochs, latent_dim, Tensor, batch_size, saving_interval)

        
        # =================================================================data_load.subject_independent(0, normalize = True)============
        #     Generate samples from trained model
        # =====================================================zero_grad========================
        no_samples_to_generate = 2*len(train_loader.dataset)
        
        if gan_type == 'vae':
            generated_data.append(np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2])))
            for ii, (edata, _) in enumerate(train_loader):
                e_data = Variable(edata.type(Tensor))
                recon, mu, logvar = vae_model(e_data)
                generated_data[target][batch_size*ii:batch_size*ii+len(e_data), :, :, :] = recon.cpu().data.numpy()
        else:
            generated_data.append(np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2])))
            for ii in range(0, no_samples_to_generate, batch_size):
                z = Variable(Tensor(np.random.normal(0, 1, (min(batch_size, no_samples_to_generate-ii) , latent_dim))))
                generated_data[target][ii:min(ii+batch_size, no_samples_to_generate), :, :, :] = generator(z).cpu().data.numpy()
            del generator, discriminator

        del train_loader, train_dat, y_train, Tensor
        
            
    
    x_train = np.concatenate((generated_data[0], 
                              generated_data[1],
                              np.expand_dims(data[sub]['xtrain'][:, :, :76], axis=1)))
    y_train = np.concatenate((np.zeros((no_samples_to_generate,)), 
                              np.ones((no_samples_to_generate,)),
                              data[sub]['ytrain']))
    # =============================================================================
    #     Data augmentation and testing
    # =============================================================================
    # x_test = np.concatenate((data[sub]['xtrain'][data[sub]['ytrain'] == 0][144:288,:,:image_shape[2]], 
    #                          data[sub]['xtrain'][data[sub]['ytrain'] == 1][144:288,:,:image_shape[2]]))
    # x_test = np.expand_dims(x_test, axis=1)
    # y_test = np.concatenate((np.zeros((144,)), np.ones((144,))))
    x_test, y_test = test_data[sub]['xtest'][:,:,:76], test_data[sub]['ytest']
    x_test = np.expand_dims(x_test, axis=1)
    
    print('Train data shape: ', x_train.shape, 'Test data shape: ', x_test.shape, 'Generated data shape: ', generated_data[0].shape[0]+generated_data[1].shape[0])
    
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
    
    train_tensor = TensorDataset(x_train, y_train.type(dtype = torch.long))
    train_dataloader = DataLoader(train_tensor, batch_size = batch_size, shuffle = True)
    test_tensor = TensorDataset(x_test, y_test.type(dtype = torch.long))
    test_dataloader = DataLoader(test_tensor, batch_size = batch_size, shuffle = False)
    
    model = CNN(image_shape)
    accuracy = train_cnn_model(model, train_dataloader, test_dataloader, epochs=300)
    print("-"*20)
    print("Accuracy: " + str(accuracy) + " sub: " + str(sub))
    accuracies.append(accuracy)
    del model, x_train, x_test, y_train, y_test, train_dataloader, train_tensor, test_dataloader, test_tensor
    torch.cuda.empty_cache()

del data    
with open(gan_type+'-subject-specific-results.txt', 'w') as f:
    f.write(str(accuracies))
