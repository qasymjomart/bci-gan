#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:39:09 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

AUGMENTATION: make Subject-independent augmentation and classification

"""

import numpy as np
from argparse import ArgumentParser

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch
from torchvision import transforms
torch.cuda.empty_cache()

from data_import import Data_loader, CustomEEGDataset
from cnn import CNN, train_cnn_model
cuda = True if torch.cuda.is_available() else False
# if cuda has runtime error, then run "nvidia-modprobe -u" in terminal (first download nvidia-modprobe in ubuntu)
print('cuda: ', cuda)
torch.manual_seed(0)
torch.backends.cudnn.backends = True

#%%

# Data parameters
latent_dim = 100
height = 16
width = 76
depth = 1
image_shape = (depth, height, width)

# GAN Training parameters
argparse = ArgumentParser()
argparse.add_argument("--gan_type", default='dcgan', type=str)
args = argparse.parse_args()

gan_type = args.gan_type
if gan_type == 'wgan_gp':
	from wgan_gp import Generator, Discriminator, train_model
elif gan_type == 'dcgan':
	from dcgan import Generator, Discriminator, train_model, weights_init_normal
elif gan_type == 'vae':
	from vae import VAE, train_model

batch_size = 32
lr = 0.0001
num_epochs= 500
lambda_gp = 10
n_discriminator = 5
saving_interval = num_epochs/10
accuracies = {}
all_accuracies={}
#%% 
normalize = transforms.Normalize([0.5], [0.5])
sub_idxs = [0,1,2,3,4,5,6,7,8,9]
data_load = Data_loader(dataset_name = 'TenHealthyData')

#%%
sample_sizes = [20,50,100,150,200,250,288]

for sample_size in sample_sizes:

	all_accuracies[str(sample_size)] = []

	for ratio_generate in [1,2,4]:
		accuracies[str(ratio_generate)] = []

	for sub in sub_idxs:
		
		data, test_data = data_load.subject_independent(sub, sample_size = sample_size, normalize = True)

		generated_data = []
		generator_models = []
		for target in [0, 1]: # generate Target and Nontarget samples
			# Data preparation
			gan_train_data = np.array(data['xtrain'][data['ytrain'] == target])	
			
			# Moving data to torch
			gan_train_data = torch.unsqueeze(torch.from_numpy(gan_train_data), axis=1)
			y_train = torch.ones((gan_train_data.shape[0],))
			
			train_dat = CustomEEGDataset(gan_train_data, y_train.type(dtype = torch.long), transform = normalize)
			train_loader = DataLoader(train_dat, batch_size = batch_size, shuffle = True)
			# =============================================================================
			#     GAN Model initialization
			# =============================================================================
			
			# Depending on type of GAN, make training:
			if gan_type == 'dcgan':
				adversarial_loss = torch.nn.BCELoss()
				
				generator = Generator(latent_dim, image_shape)
				discriminator = Discriminator(latent_dim, image_shape)

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
				
				generator_models.append(generator)
				
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
							Tensor, batch_size, saving_interval, plotting=False)
				generator_models.append(generator)
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
										vae_optimizer, vae_loss, num_epochs, latent_dim, Tensor, batch_size, saving_interval, plotting=False)
				generator_models.append(vae_model)
			# ==============================accuracies[str(ratio_generate)] = []===============================================
			#     Generate samples from trained model
			# =============================================================================

		del train_dat, y_train

		for ratio_generate in [1, 2, 4]:
			
			no_samples_to_generate = ratio_generate*len(train_loader.dataset)
			print('No. of samples to generate for this class:', no_samples_to_generate)
			generated_data = []
			for target in [0, 1]:

				if gan_type == 'vae':
					generated_data.append(np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2])))
					for jj in range(0, no_samples_to_generate, len(train_loader.dataset)):
						for ii, (edata,_) in enumerate(train_loader):
							e_data = Variable(edata.type(Tensor))
							recon, mu, logvar = generator_models[target](e_data)
							generated_data[target][jj+(ii*batch_size):jj+(ii*batch_size+len(e_data)), :, :, :] = recon.mul_(0.5).add_(0.5).cpu().data.numpy()
				else:
					generated_data.append(np.empty((no_samples_to_generate, image_shape[0], image_shape[1], image_shape[2])))
					for ii in range(0, no_samples_to_generate, batch_size):
						z = Variable(Tensor(np.random.normal(0, 1, (min(batch_size, no_samples_to_generate-ii) , latent_dim))))
						generated_data[target][ii:min(ii+batch_size, no_samples_to_generate), :, :, :] = generator_models[target](z).mul_(0.5).add_(0.5).cpu().data.numpy()

				
		
			x_train = np.concatenate((generated_data[0], 
									  generated_data[1],
									  np.expand_dims(data['xtrain'], axis=1)))
			y_train = np.concatenate((np.zeros((no_samples_to_generate,)), 
									  np.ones((no_samples_to_generate,)),
									  data['ytrain']))
			# =============================================================================
			#     Data augmentation and testing
			# =============================================================================
			x_test = np.expand_dims(test_data['xtest'], axis=1)
			y_test = test_data['ytest']
			print('Train data shape: ', x_train.shape, 
				  'Test data shape: ', x_test.shape, 
				  'Generated data shape: ', generated_data[0].shape[0]+generated_data[1].shape[0])
		
			# Convert to torch from numpy
			x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
			x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
			
			train_tensor = CustomEEGDataset(x_train, y_train.type(dtype = torch.long), transform=normalize)
			train_dataloader = DataLoader(train_tensor, batch_size = batch_size, shuffle = True)
			test_tensor = CustomEEGDataset(x_test, y_test.type(dtype = torch.long), transform=normalize)
			test_dataloader = DataLoader(test_tensor, batch_size = batch_size, shuffle = False)
			
			model = CNN(image_shape)
			accuracy = train_cnn_model(model, train_dataloader, test_dataloader, epochs=300)

			print("Accuracy: " + str(accuracy) + ' ratio: ' + str(ratio_generate) + " sub: " + str(sub))
			accuracies[str(ratio_generate)].append(accuracy)

			del model, x_train, x_test, y_train, y_test, train_dataloader, train_tensor, test_dataloader, test_tensor, generated_data
	
	torch.cuda.empty_cache()
		
	with open(gan_type+'_ratio_'+str(ratio_generate)+'_sample-size_'+str(sample_size)+'_subject-independent-results.txt', 'w') as f:
		f.write(str(accuracies))

	
	all_accuracies[str(sample_size)].append(accuracies)
	del data 
with open(gan_type+'_subject_independent_results_ALL.txt', 'w') as f:
	f.write(str(all_accuracies))
