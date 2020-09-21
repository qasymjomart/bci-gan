#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:36:05 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import torch.nn as nn
import torch.autograd as autograd
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()     
        
        self.latent_dim = latent_dim
        self.image_shape= image_shape

        self.linear = nn.Sequential(
            nn.Linear(self.latent_dim, 32*2*19)
        )
        
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, self.image_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(x.shape[0], 32, 2, 19)
        b = self.conv_layers(x)
        return b


class Discriminator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Discriminator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.image_shape[0], 32, bn=True),
            *discriminator_block(32, 16, bn=True),
            *discriminator_block(16, 16, bn=True),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
                nn.Linear(16*1*10, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        b = self.fc(out)
        return b

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_model(train_loader, generator, discriminator, optimizer_generator, optimizer_discriminator, adversarial_loss, num_epochs, latent_dim, Tensor, batch_size = 32, saving_interval = 50):

    for epoch in range(num_epochs):
        for i, (edata, _) in enumerate(train_loader):
            
            real_labels = Variable(Tensor(edata.shape[0], 1).fill_(1.0), requires_grad=False)
            fake_labels = Variable(Tensor(edata.shape[0], 1).fill_(0.0), requires_grad=False)
    
            # Configure input
            real_images = Variable(edata.type(Tensor))
    
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_generator.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (real_images.shape[0], latent_dim))))
            fake_images = generator(z)
            g_loss = adversarial_loss(discriminator(fake_images), real_labels)
            g_loss.backward()
            optimizer_generator.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_discriminator.zero_grad()    
            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images.detach())
            d_loss = (adversarial_loss(real_validity, real_labels) + adversarial_loss(fake_validity, fake_labels))/2
            d_loss.backward()
            optimizer_discriminator.step()
    
 

        if epoch % saving_interval == 0:
            # save_image(fake_images.data[:25], "wgan_gp_generated_%d.png" % epoch, nrow=5, normalize=False)
            # save_image(real_images.data[:25], "wgan_gp_real_%d.png" % epoch, nrow=5, normalize=False)
            
            print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, num_epochs, d_loss.item(), g_loss.item())
            )
            
            r, c = 4,4
            fig,axarr = plt.subplots(r,c)
            cnt = 0
            for ii in range(r):
                for jj in range(c):
                    axarr[ii,jj].imshow(fake_images[cnt, 0,:,:].cpu().data.numpy())
                    axarr[ii,jj].axis('off')
                    cnt += 1
            fig.savefig("dcgan_generated_%d.png" % epoch)
            plt.close()
            
            r, c = 4,4
            fig,axarr = plt.subplots(r,c)
            cnt = 0
            for ii in range(r):
                for jj in range(c):
                    axarr[ii,jj].imshow(real_images[cnt, 0,:,:].cpu().data.numpy())
                    axarr[ii,jj].axis('off')
                    cnt += 1
            fig.savefig("dcgan_real_%d.png" % epoch)
            plt.close()
                
                
    return generator, discriminator