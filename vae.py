#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:35:34 2020

@author: Kassymzhomart aka @qasymjomart
"""

import torch
from torch import nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		
		self.latent_dim = 100
		self.image_shape= (1,16,76)
		
		def discriminator_block(in_filters, out_filters, bn=True):
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block
		
		self.encoder = nn.Sequential(
			
			*discriminator_block(self.image_shape[0], 32, bn=False),
			*discriminator_block(32, 32, bn=False),
			*discriminator_block(32, 32, bn=False)
			)
		
		self.fc1 = nn.Linear(32*2*10, self.latent_dim)
		self.fc2 = nn.Linear(32*2*10, self.latent_dim)
		self.fc3 = nn.Linear(self.latent_dim, 32*4*19)
		
		self.decoder = nn.Sequential(
			
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			
			nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			
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
		
	def bottleneck(self, x):
		mu, logvar = self.fc1(x), self.fc2(x)
		z = self.reparameterize(mu, logvar)
		
		return z, mu, logvar
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn(*mu.size())
		
		device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		std, eps = std.to(device), eps.to(device)
		
		return mu + std * eps
		
	def forward(self, input):
		x = self.encoder(input)
		x = torch.flatten(x, start_dim=1)
		z, mu, logvar = self.bottleneck(x)
		x = self.fc3(z)
		x = x.view(x.shape[0], 32, 4, 19)
		return self.decoder(x), mu, logvar
		
def plot_losses(losses):
	plt.figure(figsize=(10,5))
	plt.title("VAE loss during training")
	plt.plot(losses, label="VAE")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig("vae_loss.png")
	plt.close()


def train_model(train_loader, model, 
				optimizer, vae_loss, 
				num_epochs, latent_dim, Tensor, batch_size = 32, saving_interval = 50, plotting=True):
	
	losses = []
	device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

	for epoch in range(num_epochs):
		for i, (edata, _) in enumerate(train_loader):
	
			# Configure input
			real_data = edata.to(device)
	
			# ---------------------
			#  Train Discriminator
			# ---------------------
			# Train discriminator n_discriminator times
			optimizer.zero_grad()
		
			# Generate a batch of images
			recon_data, mu, logvar = model(real_data)
	
			loss = vae_loss(recon_data, real_data) - 0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
			
			loss.backward()
			
			optimizer.step()

	
		losses.append(loss.item())
		
		if epoch % saving_interval == 0:
			# save_image(fake_images.data[:25], "wgan_gp_generated_%d.png" % epoch, nrow=5, normalize=False)
			# save_image(real_images.data[:25], "wgan_gp_real_%d.png" % epoch, nrow=5, normalize=False)
			
			print(
			"[Epoch %d/%d] [loss: %f]"
			% (epoch, num_epochs, loss.item())
			)
			
			if plotting:
				r, c = 4,4
				fig,axarr = plt.subplots(r,c)
				cnt = 0
				for ii in range(r):
					for jj in range(c):
						axarr[ii,jj].imshow(recon_data[cnt, 0,:,:].cpu().data.numpy())
						axarr[ii,jj].axis('off')
						cnt += 1
				fig.savefig("vae_reconstructed_%d.png" % epoch)
				plt.close()
				
				r, c = 4,4
				fig,axarr = plt.subplots(r,c)
				cnt = 0
				for ii in range(r):
					for jj in range(c):
						axarr[ii,jj].imshow(real_data[cnt, 0,:,:].cpu().data.numpy())
						axarr[ii,jj].axis('off')
						cnt += 1
				fig.savefig("vae_real_%d.png" % epoch)
				plt.close()
					
	plot_losses(losses)              
	return model
		
		
	
	
	