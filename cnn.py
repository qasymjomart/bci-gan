#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:36:41 2020

@author: bigdata
"""

import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        
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
    
def train_cnn_model(model, train_loader, test_loader, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
        
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            
            inputs, labels = inputs.type(Tensor).to(device), labels.to(device)
    
            # zero the parameter gradients
            cnn_optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
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
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy