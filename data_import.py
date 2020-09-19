#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:28:49 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

"""

import mne
import pickle
import numpy as np
# from sklearn.model_selection import train_test_split

# import itertools
# import time, copy, pdb
# import pandas as pd 

import os

# sklearn standard scaler  
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

def data_pooler(dataset_name, augment=True):
	'''
		This function intakes a dataset with a certain name and imports it.

		Input: 
		- dataset_name

		Return:
		- ss1
	'''

	filename = dataset_name + '.pickle'
	with open(filename, 'rb') as fh:
		d1 = pickle.load(fh)

	ss1 = []
	for ii in range(len(d1)):
		ss1.append(subject_specific([ii], d1, augment))

	return ss1

def subject_specific(subjectIndex, d1, augment):
    # returns torch tensors with extract positive and negative classes  
    """Leave one subject out wrapper        
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


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
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