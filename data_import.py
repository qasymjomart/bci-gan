#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:28:49 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

"""

import mne
import pickle
import numpy as np

# import itertools
# import time, copy, pdb
# import pandas as pd 

# sklearn standard scaler  
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Data_loader():
	def __init__(self, dataset_name):
		self.dataset_name = dataset_name

	def pool_one_class(self, classtype = 'Target', normalize = True):
		'''
			This function intakes a dataset with a certain name and imports one class data only.
	
			Input: 
			- classtype
			- normalize
	
			Return:
			- data
		'''
	
		filename = self.dataset_name + '.pickle'
		with open(filename, 'rb') as fh:
			d1 = pickle.load(fh)
	
		ss1 = []
		for ii in range(len(d1)):
			ss1.append(subject_specific([ii], d1))
		
		classtype = 1 if classtype == 1 else 0
		data = []

		for ii in range(len(ss1)):
			if ii == 0:
				data = np.array(ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == classtype][:288, :, :76])
			else:
				data = np.concatenate((data, ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == classtype][:288, :, :76]))
		
		if normalize:
# 			data = MinMaxNormalization(data)
			scaler = NDStandardScaler()
			data = scaler.fit_transform(data)
		
		return data
	
	def pool_all_data(self, normalize=True):
		'''
			This function intakes a dataset with a certain name and pools all data, by dividing into train and test data 
			with 20% ratio.
	
			Input: 
			- classtype
			- normalize
	
			Return:
			- data
		'''
		filename = self.dataset_name + '.pickle'
		with open(filename, 'rb') as fh:
			d1 = pickle.load(fh)
	
		ss1 = []
		for ii in range(len(d1)):
			ss1.append(subject_specific([ii], d1))
		
		for ii in range(len(ss1)):
			if ii==0:
				X = ss1[ii][0]['xtrain'][:, :, :76]
				Y = ss1[ii][0]['ytrain'][:, :, :76]
			else:
				X = np.concatenate((X, ss1[ii][0]['xtrain'][:, :, :76]))
				Y = np.concatenate((Y, ss1[ii][0]['ytrain'][:, :, :76]))
		
		if normalize:
			X = MinMaxNormalization(X)
		
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
		
		data = dict(xtrain=x_train, ytrain=y_train)
		test_data = dict(xtest=x_test, ytest=y_test)
		
		return data, test_data
		
	def subject_specific(self, normalize = True):
		'''
			This function intakes a dataset with a certain name and imports in a subject-specific way.
	
			Input: 
			- normalize
	
			Return:
			- data
		'''
	
		filename = self.dataset_name + '.pickle'
		with open(filename, 'rb') as fh:
			d1 = pickle.load(fh)
	
		ss1 = []
		for ii in range(len(d1)):
			ss1.append(subject_specific([ii], d1, augment=False))
			
		data = []
		test_data = []
		for ii in range(len(ss1)):

			X = np.concatenate((ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 1][:,:,:76], ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 0][:288, :, :76])) 
			if normalize == True:
				scaler = NDStandardScaler()
				X = scaler.fit_transform(X)
# 			Y = ss1[ii][0]['ytrain']
			Y = np.concatenate((np.ones((288,)), np.zeros((288,))))

			# indx = np.arange(288)
			# # np.random.shuffle(indx)
			# indx0 = indx[:144]

			# x_train, y_train = X[indx0], Y[indx0]
			# x_test, y_test = X[[i for i in np.arange(288) if i not in indx0]], Y[[i for i in np.arange(288) if i not in indx0]]

			# # np.random.shuffle(indx)
			# indx1 = indx[:144] + 288
			# x_train, y_train = np.concatenate((x_train, X[indx1])), np.concatenate((y_train, Y[indx1]))
			# x_test, y_test = np.concatenate((x_test, X[[i for i in np.arange(288, 576) if i not in indx1]])), np.concatenate((y_test, Y[[i for i in np.arange(288, 576) if i not in indx1]]))

			x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle=True, stratify=Y)

			data.append(dict(xtrain=x_train, ytrain=y_train))
			test_data.append(dict(xtest = x_test, ytest=y_test))
			del x_train, y_train, x_test, y_test, X, Y
			
		return data, test_data
	
	def subject_independent(self, test_sub, sample_size = 288, normalize = True):
		'''
			This function intakes a dataset with a certain name and imports in a subject-independent way.
	
			Input: 
			- test_sub - test sub index 
			- normalize
	
			Return:
			- data
			- test_data
		'''
		filename = self.dataset_name + '.pickle'
		with open(filename, 'rb') as fh:
			d1 = pickle.load(fh)
	
		ss1 = []
		for ii in range(len(d1)):
			ss1.append(subject_specific([ii], d1))
			
		data = []
		X, Y = None, None
		x_test = np.concatenate((ss1[test_sub][0]['xtrain'][ss1[test_sub][0]['ytrain'] == 1][:,:,:76], ss1[test_sub][0]['xtrain'][ss1[test_sub][0]['ytrain'] == 0][:288,:,:76])) 
		y_test = np.concatenate((np.ones((288,)), np.zeros((288,))))
		for ii in range(len(ss1)):
			if ii == test_sub:
				pass
			else:
				indx = np.arange(288)
				np.random.shuffle(indx)
				indx0 = indx[:sample_size//2]
				np.random.shuffle(indx)
				indx1 = indx[:sample_size//2]
				try:
					temp_X0 = ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 0][indx0, :, :76]
					temp_X1 = ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 1][indx1, :, :76]
					temp_X = np.concatenate((temp_X0, temp_X1))
					temp_Y0 = ss1[ii][0]['ytrain'][ss1[ii][0]['ytrain'] == 0][indx0]
					temp_Y1 = ss1[ii][0]['ytrain'][ss1[ii][0]['ytrain'] == 1][indx1]
					temp_Y = np.concatenate((temp_Y0, temp_Y1))
					X = np.concatenate((X, temp_X))
					Y = np.concatenate((Y, temp_Y))
				except:
					temp_X0 = ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 0][indx0, :, :76]
					temp_X1 = ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 1][indx1, :, :76]
					X = np.concatenate((temp_X0, temp_X1))
					temp_Y0 = ss1[ii][0]['ytrain'][ss1[ii][0]['ytrain'] == 0][indx0]
					temp_Y1 = ss1[ii][0]['ytrain'][ss1[ii][0]['ytrain'] == 1][indx1]
					Y = np.concatenate((temp_Y0, temp_Y1))
					
		if normalize:
			scaler = NDStandardScaler()
			X = scaler.fit_transform(X)
			x_test = scaler.transform(x_test)

		test_data = dict(xtest = x_test, ytest = y_test)
		data = dict(xtrain=X, ytrain=Y)
		
		return data, test_data
		
def subject_specific(subjectIndex, d1, augment=False):
	"""      
	Input: d1 - is list consisting of subject-specific epochs in MNE structure
	Example usage:
		subjectIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]            
		for sub in subjectIndex:
			xvalid = subject_specific(sub, d1)          
	"""
	pos_str = 'Target'
	neg_str = 'NonTarget'

	data, pos, neg = [], [] , []    
	if len(subjectIndex) > 1: # multiple subjects             
		for jj in subjectIndex:                
			print('Loading subjects:', jj)   
			dat = d1[jj]                                     
			pos.append(dat[pos_str].get_data())
			neg.append(dat[neg_str].get_data())  
	else: 
		print('Loading subject:', subjectIndex[0])  
		dat = d1[subjectIndex[0]]
		pos.append(dat[pos_str].get_data())
		neg.append(dat[neg_str].get_data())
	
	if augment == True:
		for ii in range(len(pos)):
			# subject specific upsampling of minority class 
			targets = pos[ii]              
			for j in range((neg[ii].shape[0]//pos[ii].shape[0]) - 1): 
				targets = np.concatenate([pos[ii], targets])                    
			pos[ii] = targets  
	
	for ii in range(len(pos)):            
		X = np.concatenate([pos[ii].astype('float32'), neg[ii].astype('float32')])            
		Y = np.concatenate([np.ones(pos[ii].shape[0]).astype('float32'), 
							np.zeros(neg[ii].shape[0]).astype('float32')])       
	data.append(dict(xtrain = X, ytrain = Y))            
	return data

def MinMaxNormalization(data):
	return (data - np.min(data))/(np.max(data) - np.min(data))

class NDStandardScaler(TransformerMixin):
	def __init__(self, **kwargs):
		self._scaler = MinMaxScaler(copy=True, **kwargs)
		self._orig_shape = None

	def fit(self, X, **kwargs):
		X = np.array(X)
		# Save the original shape to reshape the flattened X later
		# back to its original shape
		if len(X.shape) > 1:
			self._orig_shape = X.shape[1:]
		X = self._flatten(X)
		self._scaler.fit(X, **kwargs)
		return self

	def transform(self, X, **kwargs):
		X = np.array(X)
		X = self._flatten(X)
		X = self._scaler.transform(X, **kwargs)
		X = self._reshape(X)
		return X

	def _flatten(self, X):
		# Reshape X to <= 2 dimensions
		if len(X.shape) > 2:
			n_dims = np.prod(self._orig_shape)
			X = X.reshape(-1, n_dims)
		return X

	def _reshape(self, X):
		# Reshape X back to it's original shape
		if len(X.shape) >= 2:
			X = X.reshape(-1, *self._orig_shape)
		return X

class CustomEEGDataset(Dataset):
	"""EEG train Custom dataset."""

	def __init__(self, data, labels, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = {'xtrain': data, 'ytrain': labels}
        
		self.transform = transform

	def __len__(self):
		return len(self.data['xtrain'])

	def __getitem__(self, idx):
		try:
			sample = self.data['xtrain'][idx]
			label = self.data['ytrain'][idx]
			if self.transform:
				sample = self.transform(sample)
			return sample, label
		except:
			print("EEG Data is not imported yet. Please first import subjects using import_subjects method.")

class CustomEEGDataset_OnlyData(Dataset):
	"""EEG train Custom dataset."""

	def __init__(self, data, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = {'xtrain': data}
        
		self.transform = transform

	def __len__(self):
		return len(self.data['xtrain'])

	def __getitem__(self, idx):
		try:
			sample = self.data['xtrain'][idx]

			if self.transform:
				sample = self.transform(sample)

			return sample
		except:
			print("EEG Data is not imported yet. Please first import subjects using import_subjects method.")

class Reshape(object):
    """Rescale the image in a sample to a given size using duplicating

    Args:
        output_size (tuple or int):
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = int(self.output_size), int(self.output_size)


        img = self.pad_by_duplicating(sample, new_h, new_w)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        img = img[None, :, :] * np.ones(3, dtype=int)[:, None, None]
        return torch.from_numpy(img).double()
    
    def pad_by_duplicating(self, x, desired_height=224, desired_width=224): #duplicate signal until the desired new array is full
        x_height, x_width = x.shape[0], x.shape[1]
        new_x = np.zeros((desired_height, desired_width))
        for nhx in range(0, desired_height, x_height):
            for nwx in range(0, desired_width, x_width):
                new_x[nhx:min(nhx+x_height, desired_height), nwx:min(nwx+x_width, desired_width)] = x[0:min(x_height, desired_height-nhx), 0:min(x_width, desired_width-nwx)]
        return new_x 