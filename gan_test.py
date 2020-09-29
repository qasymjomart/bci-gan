#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:48:53 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("bright", 4)

from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import torch.nn as nn
import torch.autograd as autograd
import torch


def gan_test(real_data, generated_data, classifier):
    """

    Parameters
    ----------
    generated_data : DICTIONARY (for X & Y)
        Generated (by GAN) EEG data to be used as training data
    real_data : DICTIONARY (for X & Y)
        Real EEG data to be used as test data.
    classifier : STRING
        Name of the classifier to be used for GAN-test

    Returns
    -------
    accuracy : FLOAT NUMBER
        accuracy of the classifer, trained on generated data and testes on real data.

    """
    
    x_train, y_train = shuffle(real_data['x'], real_data['y'])
    x_test, y_test = shuffle(generated_data['x'], generated_data['y'])
    
    accuracy = 0
    
    if classifier == 'LDA':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    elif classifier == 'LogisticRegression':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    elif classifier == 'SVM':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
        clf = LinearSVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    elif classifier == 'DCNN':
        x_train = torch.unsqueeze(torch.from_numpy(x_train), axis=1)
        train_dat = TensorDataset(x_train, torch.from_numpy(y_train).type(dtype = torch.long))
        train_loader = DataLoader(train_dat, batch_size = 32, shuffle = True)
        
        x_test = torch.unsqueeze(torch.from_numpy(x_test), axis=1)
        test_dat = TensorDataset(x_test, torch.from_numpy(y_test).type(dtype = torch.long))
        test_loader = DataLoader(test_dat, batch_size = 32, shuffle = False)
        
        
        cnn = DCNN(x_train[0].shape)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        cnn.to(device)
            
        criterion = nn.CrossEntropyLoss()
        cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
        
        for epoch in range(50):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                
                inputs, labels = inputs.type(Tensor).to(device), labels.to(device)
        
                # zero the parameter gradients
                cnn_optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                cnn_optimizer.step()
        
                # print statistics
                running_loss += loss.item()
            print('Epoch %d, loss: %.3f' %
                          (epoch + 1, running_loss / len(train_loader)))
            running_loss = 0.0
        
        print('Finished Training')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.type(Tensor).to(device), labels.to(device)
                outputs = cnn(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network: %d %%' % (
            100 * correct / total))
        
        accuracy = 100 * correct / total
    return accuracy


class DCNN(nn.Module):
    def __init__(self, input_shape):
        super(DCNN, self).__init__()
        
        self.input_shape = input_shape
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 1, 1), nn.MaxPool2d(2), nn.ReLU(inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.input_shape[0], 64, bn=False),
            *discriminator_block(64, 32, bn=False),
            nn.Dropout(0.2)
        )

        self.fc = nn.Sequential(
                nn.Linear(32*4*19, 2),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        b = self.fc(out)
        return b
    
#%% t-SNE Visualisation

def t_sne(real_data, generated_data, gan_type):
    """
    Calculates t-SNE for data and plots its

    Parameters
    ----------
    real_data : DICT
        Real EEG data given as DICT ['x','y'].
    generated_data : DICT
        Generated EEG data given as DICT ['x','y'].

    Returns
    -------
    sns_plot : SNS PLOT
        Seaborn plot with t-SNE mapped.

    """
    
    x_gen, y_gen = generated_data['x'], generated_data['y']
    x_real, y_real = real_data['x'], real_data['y']
    
    for ii in range(y_gen.shape[0]):
        if y_gen[ii] == 0:
            y_gen[ii] = 0
        elif y_gen[ii] == 1:
            y_gen[ii] = 1
    
    for ii in range(y_real.shape[0]):
        if y_real[ii] == 0:
            y_real[ii] = 2
        elif y_real[ii] == 1:
            y_real[ii] = 3
    # reshape data to (n_samples, n_features)
    x_gen, x_real = np.reshape(x_gen, (x_gen.shape[0], x_gen.shape[1]*x_gen.shape[2])), np.reshape(x_real, (x_real.shape[0], x_real.shape[1]*x_real.shape[2]))
    
    # concatenate all to one x
    X, y = np.concatenate((x_gen, x_real)), np.concatenate((y_gen, y_real))
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    
    palette = sns.color_palette("bright", 4)
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, palette=palette)
    new_labels = ['NonTarget Gen', 'Target Gen', 'NonTarget Real', 'Target Real']
    for t, l in zip(sns_plot.legend().texts, new_labels): t.set_text(l)
    fig = sns_plot.get_figure()
    fig.savefig(gan_type+'_tSNE.png')
    return sns_plot

def t_sne_one_data(real_data):
    """
     The same as t_sne, but only for one data

    Parameters
    ----------
    real_data : DICT
        EEG data given as DICT ['x','y'].

    Returns
    -------
    sns_plot : SNS
        Seabron plot with t_SNE mapped.

    """
    
    x_real, y_real = real_data['x'], real_data['y']
    print('Starting fitting t-SNE')
    
    # reshape data to (n_samples, n_features)
    x_real = np.reshape(x_real, (x_real.shape[0], x_real.shape[1]*x_real.shape[2]))
    
    # concatenate all to one x
    X, y = x_real, y_real
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    print('Finished fitting t-SNE.')
    palette = sns.color_palette("bright", 2)
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, palette=palette)
    new_labels = ['NonTarget', 'Target']
    for t, l in zip(sns_plot.legend().texts, new_labels): t.set_text(l)
    fig = sns_plot.get_figure()
    fig.savefig('tSNE_one_data.png')
    return sns_plot

#%% KL divergence



