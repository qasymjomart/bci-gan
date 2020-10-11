#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:39:09 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

AUGMENTATION: make Subject-independent augmentation and classification

"""

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch
torch.cuda.empty_cache()

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
gan_type = 'dcgan'
if gan_type == 'wgan_gp':
	from wgan_gp import Generator, Discriminator, train_model
elif gan_type == 'dcgan':
	from dcgan import Generator, Discriminator, train_model, weights_init_normal
elif gan_type == 'vae':
	from vae import VAE, train_model

batch_size = 32
lr = 0.0001
num_epochs= 100
lambda_gp = 10
n_discriminator = 5
saving_interval = num_epochs/10
accuracies = {}

#%% 

sub_idxs = [0,1,2,3,4,5,6,7,8,9]
data_load = Data_loader(dataset_name = 'TenHealthyData')

#%%
sample_sizes = [20, 50, 100, 150, 200, 250, 288]

for sample_size in sample_sizes:
	accuracies[str(sample_size)] = []
	for sub in sub_idxs:
		
		data, test_data = data_load.subject_independent(sub, sample_size = sample_size, normalize = True)
		no_samples_to_generate = data['xtrain'].shape[0]//2
		generated_data = []
		for target in [0, 1]: # generate Target and Nontarget samples
			# Data preparation
			gan_train_data = np.array(data['xtrain'][data['ytrain'] == target])	
			
			# Moving data to torch
			gan_train_data = torch.unsqueeze(torch.from_numpy(gan_train_data), axis=1)[:,:,:,:image_shape[2]]
			y_train = torch.ones((gan_train_data.shape[0],))
			
			train_dat = TensorDataset(gan_train_data, y_train.type(dtype = torch.long))
			train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)
			# =============================================================================
			#     GAN Model initialization
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
							Tensor, batch_size, saving_interval, plotting=False)     
				
				
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
							Tensor, batch_size, saving_interval, plotting=False)
			
			elif gan_type == 'vae':
						
				vae_model = VAE()
				vae_loss = torch.nn.BCELoss()
				
				if cuda:
					vae_model.cuda()
					vae_loss.cuda()
					
				vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
				
				Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

				# =============================================================================
				#     Model Training
				# =============================================================================
				
				vae_model = train_model(train_loader, vae_model, 
										vae_optimizer, vae_loss, num_epochs, latent_dim, Tensor, batch_size, saving_interval, plotting=False)

			# =============================================================================
			#     Generate samples from trained model
			# =============================================================================

			no_samples_to_generate = len(train_loader.dataset)
			print(no_samples_to_generate)
			if gan_type == 'vae':
				generated_data.append(np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2])))
				for ii, (edata, _) in enumerate(train_loader):
					e_data = Variable(edata.type(Tensor))
					recon, mu, logvar = vae_model(e_data)
					generated_data[batch_size*ii:batch_size*ii+len(e_data), :, :, :] = recon.cpu().data.numpy()
				del vae_model
			else:
				generated_data.append(np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2])))
				for ii in range(0, no_samples_to_generate, batch_size):
					z = Variable(Tensor(np.random.normal(0, 1, (min(batch_size, no_samples_to_generate-ii) , latent_dim))))
					generated_data[target][ii:min(ii+batch_size, no_samples_to_generate), :, :, :] = generator(z).cpu().data.numpy()
				del generator, discriminator

			del train_loader, train_dat, y_train, Tensor
		
		x_train = np.concatenate((generated_data[0], 
								  generated_data[1],
								  np.expand_dims(data['xtrain'][data['ytrain'] == 0], axis=1)[:,:,:,:76],
								  np.expand_dims(data['xtrain'][data['ytrain'] == 1], axis=1)[:,:,:,:76]))
		y_train = np.concatenate((np.zeros((no_samples_to_generate,)), 
								  np.ones((no_samples_to_generate,)),
								  data['ytrain'][data['ytrain'] == 0], 
								  data['ytrain'][data['ytrain'] == 1]))
		# =============================================================================
		#     Data augmentation and testing
		# =============================================================================
		x_test = np.expand_dims(test_data['xtest'], axis=1)[:,:,:,:76]
		y_test = test_data['ytest']
		print('Train data shape: ', x_train.shape, 'Test data shape: ', x_test.shape, 'Generated data shape: ', generated_data[0].shape[0]+generated_data[1].shape[0])
    
		# Convert to torch from numpy
		x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
		x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
		
		train_tensor = TensorDataset(x_train, y_train.type(dtype = torch.long))
		train_dataloader = DataLoader(train_tensor, batch_size = batch_size, shuffle = True)
		test_tensor = TensorDataset(x_test, y_test.type(dtype = torch.long))
		test_dataloader = DataLoader(test_tensor, batch_size = batch_size, shuffle = False)
		
		model = CNN(image_shape)
		accuracy = train_cnn_model(model, train_dataloader, test_dataloader, epochs=100)

		print("Accuracy: " + str(accuracy) + " sub: " + str(sub))
		accuracies[str(sample_size)].append(accuracy)

		del model, x_train, x_test, y_train, y_test, train_dataloader, train_tensor, test_dataloader, test_tensor
		torch.cuda.empty_cache()
	
	del data    
	with open(gan_type+'_sample-size_'+str(sample_size)+'_subject-independent-results.txt', 'w') as f:
		   f.write(str(accuracies[str(sample_size)]))
   
with open(gan_type+'-subject-independent-results_ALL.txt', 'w') as f:
	f.write(str(accuracies))
