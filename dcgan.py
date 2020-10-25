#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:36:05 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

Credits are given to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
that was valuable and helpful in implementing

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision import transforms

import torch.nn as nn
import torch.autograd as autograd
import torch

class Generator(nn.Module):
	def __init__(self, latent_dim, image_shape):
		super(Generator, self).__init__()     
		
		self.latent_dim = latent_dim
		self.image_shape= image_shape

		self.linear = nn.Sequential(
			nn.Linear(self.latent_dim, 32*4*19)
		)
		
		self.conv_layers = nn.Sequential(
			
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			
			nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			
			# nn.BatchNorm2d(128),
			# nn.Upsample(scale_factor=2),
			# nn.BatchNorm2d(128),
			# nn.ReLU(),
			# nn.Conv2d(128, 128, 3, stride=1, padding=1),
			# nn.BatchNorm2d(128, 0.8),
			# nn.ReLU(),
			nn.Conv2d(32, self.image_shape[0], 3, stride=1, padding=1),
			nn.Tanh()
		)

	def forward(self, z):
		x = self.linear(z)
		x = x.view(x.shape[0], 32, 4, 19)
		b = self.conv_layers(x)
		return b


class Discriminator(nn.Module):
	def __init__(self, latent_dim, image_shape):
		super(Discriminator, self).__init__()
		
		self.latent_dim = latent_dim
		self.image_shape = image_shape
		
		def discriminator_block(in_filters, out_filters, bn=True):
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block

		self.model = nn.Sequential(
			*discriminator_block(self.image_shape[0], 32, bn=False),
			*discriminator_block(32, 32, bn=False),
			*discriminator_block(32, 32, bn=False)
			# nn.Sigmoid()
		)

		self.fc = nn.Sequential(
				nn.Linear(32*2*10, 1),
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
		
def plot_losses(g_losses, d_losses):
	plt.figure(figsize=(10,5))
	plt.title("G and D loss during training")
	plt.plot(g_losses, label="G")
	plt.plot(d_losses, label="D")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig("dcgan_loss.png")
	plt.close()
	

def train_model(train_loader, generator, discriminator, 
				optimizer_generator, optimizer_discriminator, adversarial_loss, 
				num_epochs, latent_dim, Tensor, batch_size = 32, saving_interval = 50, plotting=True):
	g_losses = []
	d_losses = []
	device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	real_label = 1.0
	fake_label = 0.0

	for epoch in range(num_epochs):
		for i, (edata, _) in enumerate(train_loader):
			# ---------------------
			#  Train Discriminator
			# ---------------------
			optimizer_discriminator.zero_grad()
			real_images = edata.to(device)
			labels = torch.full((real_images.size(0),1), real_label, dtype=torch.float, device=device)
			real_validity = discriminator(real_images)
			d_loss_real = adversarial_loss(real_validity, labels)
			d_loss_real.backward()
			d_x = real_validity.mean().item()
			
			z = Variable(Tensor(np.random.normal(0, 1, (real_images.shape[0], latent_dim))))
			labels.fill_(fake_label)
			fake_images = generator(z)
			fake_validity = discriminator(fake_images.detach())
			d_loss_fake = adversarial_loss(fake_validity, labels)
			d_loss_fake.backward()
			d_gz_1 = fake_validity.mean().item()
			d_loss = d_loss_real + d_loss_fake
			optimizer_discriminator.step()
			# -----------------
			#  Train Generator
			# -----------------
			optimizer_generator.zero_grad()
			labels.fill_(real_label)
			fake_g_validity = discriminator(fake_images)
			g_loss = adversarial_loss(fake_g_validity, labels)
			g_loss.backward()
			d_gz_2 = fake_g_validity.mean().item()
			optimizer_generator.step()      
 
		g_losses.append(g_loss.item())
		d_losses.append(d_loss.item())
			
		if epoch % saving_interval == 0:
			# save_image(fake_images.data[:25], "wgan_gp_generated_%d.png" % epoch, nrow=5, normalize=False)
			# save_image(real_images.data[:25], "wgan_gp_real_%d.png" % epoch, nrow=5, normalize=False)
			
			print(
			"[Epoch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f] [D(G(z)): %f / %f]"
			% (epoch, num_epochs, d_loss.item(), g_loss.item(), d_x, d_gz_1, d_gz_2)
			)
			
			if plotting:
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
				
	plot_losses(g_losses, d_losses)
	return generator, discriminator