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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split


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
		
		classtype = 1 if classtype == 'Target' else 0
		data = []

		for ii in range(len(ss1)):
			if ii == 0:
				data = np.array(ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == classtype][:288])
			else:
				data = np.concatenate((data, ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == classtype][:288]))
		
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
				X = ss1[ii][0]['xtrain']
				Y = ss1[ii][0]['ytrain']
			else:
				X = np.concatenate((X, ss1[ii][0]['xtrain']))
				Y = np.concatenate((Y, ss1[ii][0]['ytrain']))
		
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
		for ii in range(len(ss1)):
# 			X = ss1[ii][0]['xtrain'][:288]
			X = np.concatenate((ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 1], ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 0][:288])) 
			if normalize == True:
				X = MinMaxNormalization(X) 
# 			Y = ss1[ii][0]['ytrain']
			Y = np.concatenate((np.ones((288)), np.zeros((288,))))
			data.append(dict(xtrain=X, ytrain=Y))
			
		return data
	
	def subject_independent(self, test_sub, normalize = True):
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
		x_test = np.concatenate((ss1[test_sub][0]['xtrain'][ss1[test_sub][0]['ytrain'] == 1], ss1[test_sub][0]['xtrain'][ss1[test_sub][0]['ytrain'] == 0][:288])) 
		y_test = np.concatenate((np.ones((288,)), np.zeros((288,))))
		for ii in range(len(ss1)):
			if ii == test_sub:
				pass
			else:
				try:
					temp_X = np.concatenate((ss1[ii][0]['xtrain'][ss1[test_sub][0]['ytrain'] == 1], ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 0][:288]))
					temp_Y = np.concatenate((np.ones((288,)), np.zeros((288,))))
					X = np.concatenate((X, temp_X))
					Y = np.concatenate((Y, temp_Y))
				except:
					X = np.concatenate((ss1[ii][0]['xtrain'][ss1[test_sub][0]['ytrain'] == 1], ss1[ii][0]['xtrain'][ss1[ii][0]['ytrain'] == 0][:288]))
					Y = np.concatenate((np.ones((288,)), np.zeros((288,))))
					
		if normalize:
			Xmax, Xmin = np.max(X), np.min(X)
			x_test = (x_test - Xmin) / (Xmax - Xmin)
			X = MinMaxNormalization(X)
		
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
# 	data = []
# 	for ii in range(len(ss1)):
# 		X = ss1[ii][0]['xtrain']
# 		Y = ss1[ii][0]['ytrain']
# 		X = (X - np.min(X))/(np.max(X) - np.min(X))
# 		data.append(dict(xtrain=X, ytrain=Y))
		

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