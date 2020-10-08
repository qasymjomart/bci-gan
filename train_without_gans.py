#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:09:27 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

AUGMENTATION: make Subject-specific classification

"""

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
import torch
torch.cuda.empty_cache()

from data_import import Data_loader
from cnn import CNN, train_cnn_model
cuda = True if torch.cuda.is_available() else False
# if cuda has runtime error, then run "nvidia-modprobe -u" in terminal (first download nvidia-modprobe in ubuntu)
print('cuda: ', cuda)
torch.manual_seed(0)

#%%

# Data and training parameters
height = 16
width = 76
depth = 1
image_shape = (depth, height, width)
batch_size = 32
lr = 0.0001
num_epochs= 100
accuracies = []
train_type = 'subject-independent'

if train_type == 'subject-specific':
	no_samples= 144
	sub_idxs = [0,1,2,3,4,5,6,7,8,9]
	
	data_load = Data_loader(dataset_name = 'TenHealthyData')
	data = data_load.subject_specific(normalize = True)
	
	for sub in sub_idxs:
				
		x_train = np.concatenate((np.expand_dims(data[sub]['xtrain'][data[sub]['ytrain'] == 0][:no_samples, :, :image_shape[2]], axis=1),
								  np.expand_dims(data[sub]['xtrain'][data[sub]['ytrain'] == 1][:no_samples, :, :image_shape[2]], axis=1)))
		y_train = np.concatenate((np.zeros((no_samples,)), 
								  np.ones((no_samples,))))
		# =============================================================================
		#     Data augmentation and testing
		# =============================================================================
		x_test = np.concatenate((data[sub]['xtrain'][data[sub]['ytrain'] == 0][144:288,:,:image_shape[2]], 
								 data[sub]['xtrain'][data[sub]['ytrain'] == 1][144:288,:,:image_shape[2]]))
		x_test = np.expand_dims(x_test, axis=1)
		y_test = np.concatenate((np.zeros((144,)), np.ones((144,))))
		
		
		x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
		x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
		
		train_tensor = TensorDataset(x_train, y_train.type(dtype = torch.long))
		train_dataloader = DataLoader(train_tensor, batch_size = batch_size, shuffle = True)
		test_tensor = TensorDataset(x_test, y_test.type(dtype = torch.long))
		test_dataloader = DataLoader(test_tensor, batch_size = batch_size, shuffle = False)
		
		model = CNN(image_shape)
		accuracy = train_cnn_model(model, train_dataloader, test_dataloader, epochs=100)
		print("Accuracy: " + str(accuracy) + " sub: " + str(sub))
		accuracies.append(accuracy)
		del model, x_train, x_test, y_train, y_test, train_dataloader, train_tensor, test_dataloader, test_tensor
		torch.cuda.empty_cache()
	
	del data    
	with open('without-gans-subject-specific-results.txt', 'w') as f:
		f.write(str(accuracies))
		
elif train_type == 'subject-independent':
	sample_sizes = [20, 50, 100, 150, 200, 250, 288]
	sub_idxs = [0,1,2,3,4,5,6,7,8,9]
	data_load = Data_loader(dataset_name = 'TenHealthyData')
	accuracies = {}
	for sample_size in sample_sizes:
		accuracies[str(sample_size)] = []
		
		for sub in sub_idxs:
		
			data, test_data = data_load.subject_independent(sub, sample_size=sample_size, normalize = True)
			
			x_train = np.concatenate((np.expand_dims(data['xtrain'][data['ytrain'] == 0], axis=1)[:,:,:,:76],
									  np.expand_dims(data['xtrain'][data['ytrain'] == 1], axis=1)[:,:,:,:76]))
			y_train = np.concatenate((np.zeros((data['xtrain'].shape[0]//2,)), 
									  np.ones((data['xtrain'].shape[0]//2,))))
			# =============================================================================
			#     Data augmentation and testing
			# =============================================================================
			x_test = np.expand_dims(test_data['xtest'], axis=1)
			y_test = test_data['ytest']
			
			
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
			with open('without_GAN_sample-size_'+str(sample_size)+'_subject-independent-results.txt', 'w') as f:
			   f.write(str(accuracies[str(sample_size)]))
	
			del data    
	with open('without-gan-subject-independent-results_ALL.txt', 'w') as f:
		f.write(str(accuracies))
