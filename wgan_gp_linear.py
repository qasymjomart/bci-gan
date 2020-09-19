#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:42:43 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

Credits are given to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py


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

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(x.shape[0], *self.image_shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        b = self.model(x_flat)
        return b


def compute_gradient_penalty(D, real_samples, fake_samples, Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_model(train_loader, generator, discriminator, optimizer_generator, optimizer_discriminator, num_epochs, latent_dim, lambda_gp, n_discriminator, Tensor, batch_size = 32, saving_interval = 50):

    for epoch in range(num_epochs):
        for i, (edata, _) in enumerate(train_loader):
    
            # Configure input
            real_images = Variable(edata.type(Tensor))
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_discriminator.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (real_images.shape[0], latent_dim))))
    
            # Generate a batch of images
            fake_images = generator(z)
    
            # Real images
            real_validity = discriminator(real_images)
            # Fake images
            fake_validity = discriminator(fake_images)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data, Tensor)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    
            d_loss.backward()
            optimizer_discriminator.step()
    
            optimizer_generator.zero_grad()
    
            # Train the generator every n_discriminator steps
            if i % n_discriminator == 0:
    
                # -----------------
                #  Train Generator
                # -----------------
    
                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
    
                g_loss.backward()
                optimizer_generator.step()
    
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )
    
                if epoch % saving_interval == 0:
                    # save_image(fake_images.data[:25], "wgan_gp_generated_%d.png" % epoch, nrow=5, normalize=False)
                    # save_image(real_images.data[:25], "wgan_gp_real_%d.png" % epoch, nrow=5, normalize=False)
                    
                    r, c = 5,5
                    fig,axarr = plt.subplots(r,c)
                    cnt = 0
                    for ii in range(r):
                        for jj in range(c):
                            axarr[ii,jj].imshow(fake_images[cnt, 0,:,:].cpu().data.numpy())
                            axarr[ii,jj].axis('off')
                            cnt += 1
                    fig.savefig("wan_gp_generated_%d.png" % epoch)
                    plt.close()