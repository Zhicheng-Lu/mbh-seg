from torch import nn
import torch
from utils.PositionalEncoding import PositionalEncoding


class Segmentation(nn.Module):
	def __init__(self, data_reader):
		super(Segmentation, self).__init__()
		self.data_reader = data_reader
		f_size = data_reader.f_size
		self.patch_size = data_reader.patch_size
		self.n_patches = data_reader.height // self.patch_size
		d_model = data_reader.d_model
		n_head = data_reader.n_head
		num_layers = data_reader.num_layers
		num_classes = data_reader.num_classes

		# Convolutional layers
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=f_size, out_channels=f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(f_size),
			nn.ReLU()
		)
		self.up1 = nn.Sequential(
			nn.Conv2d(in_channels=4*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(2*f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(2*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*f_size, out_channels=f_size, kernel_size=(2,2), stride=(2,2))
		)
		self.up2 = nn.Sequential(
			nn.Conv2d(in_channels=8*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(4*f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(4*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*f_size, out_channels=2*f_size, kernel_size=(2,2), stride=(2,2))
		)
		self.up3 = nn.Sequential(
			nn.Conv2d(in_channels=16*f_size, out_channels=8*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(8*f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=8*f_size, out_channels=8*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(8*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*f_size, out_channels=4*f_size, kernel_size=(2,2), stride=(2,2))
		)
		self.up4 = nn.Sequential(
			nn.Conv2d(in_channels=32*f_size, out_channels=16*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(16*f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=16*f_size, out_channels=16*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(16*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=16*f_size, out_channels=8*f_size, kernel_size=(2,2), stride=(2,2))
		)
		self.conv_final = nn.Sequential(
			nn.Conv2d(in_channels=2*f_size, out_channels=f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=f_size, out_channels=f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(f_size),
			nn.ReLU(),
			nn.Conv2d(in_channels=f_size, out_channels=num_classes, kernel_size=(1,1))
		)

		# From transformer features to convolutional features
		self.transformer_to_conv1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=d_model, out_channels=2*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(2*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(2*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(2*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=2*f_size, out_channels=2*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(2*f_size),
			nn.ReLU()
		)
		self.transformer_to_conv2 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=d_model, out_channels=4*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(4*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(4*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=4*f_size, out_channels=4*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(4*f_size),
			nn.ReLU()
		)
		self.transformer_to_conv3 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=d_model, out_channels=8*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=8*f_size, out_channels=8*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(8*f_size),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=8*f_size, out_channels=8*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=8*f_size, out_channels=8*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(8*f_size),
			nn.ReLU()
		)
		self.transformer_to_conv4 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=d_model, out_channels=16*f_size, kernel_size=(2,2), stride=(2,2)),
			nn.Conv2d(in_channels=16*f_size, out_channels=16*f_size, kernel_size=(3,3), padding=(1,1)),
			# nn.BatchNorm2d(16*f_size),
			nn.ReLU()
		)
		self.transformer_to_conv5 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=d_model, out_channels=16*f_size, kernel_size=(2,2), stride=(2,2)),
		)

		# Transformer patch embedding & positional encoding
		self.patch_embedding = nn.Sequential(
			nn.Linear(in_features=self.patch_size*self.patch_size, out_features=d_model)
		)

		self.positional_encoding = PositionalEncoding(d_model)

		# Transformer layers
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

		# self.patches_projection = nn.Sequential(
		# 	nn.Linear(in_features=d_model, out_features=self.patch_size*self.patch_size)
		# )


	def forward(self, device, imgs): # img -> h * w * d
		h, w, d = imgs.shape

		# Transformer pre-processing
		patches = self.imgs_to_patches(device, imgs, self.n_patches, self.patch_size) # img -> seq * batch_size * (p_size*p_size)
		patches = self.patch_embedding(patches) # img -> seq * batch_size * d_model
		patches = self.positional_encoding(patches) # img -> seq * batch_size * d_model

		# Transformer layers
		transformer_features1 = self.transformer_encoder(patches) # img -> seq * batch_size * d_model
		transformer_features2 = self.transformer_encoder(transformer_features1) # img -> seq * batch_size * d_model
		transformer_features3 = self.transformer_encoder(transformer_features2) # img -> seq * batch_size * d_model
		transformer_features4 = self.transformer_encoder(transformer_features3) # img -> seq * batch_size * d_model
		transformer_features5 = self.transformer_encoder(transformer_features4) # img -> seq * batch_size * d_model

		# Layer 5
		features = self.patches_to_imgs(transformer_features5, self.n_patches)
		layer_5 = self.transformer_to_conv5(features)

		# Layer 4
		features = self.patches_to_imgs(transformer_features4, self.n_patches)
		layer_4 = self.transformer_to_conv4(features)
		layer_4 = torch.cat((layer_5, layer_4), 1)
		layer_4 = self.up4(layer_4)

		# Layer 3
		features = self.patches_to_imgs(transformer_features3, self.n_patches)
		layer_3 = self.transformer_to_conv3(features)
		layer_3 = torch.cat((layer_4, layer_3), 1)
		layer_3 = self.up3(layer_3)

		# Layer 2
		features = self.patches_to_imgs(transformer_features2, self.n_patches)
		layer_2 = self.transformer_to_conv2(features)
		layer_2 = torch.cat((layer_3, layer_2), 1)
		layer_2 = self.up2(layer_2)

		# Layer 1
		features = self.patches_to_imgs(transformer_features1, self.n_patches)
		layer_1 = self.transformer_to_conv1(features)
		layer_1 = torch.cat((layer_2, layer_1), 1)
		layer_1 = self.up1(layer_1)

		# Layer 0
		layer_0 = self.conv(torch.reshape(imgs, (d, 1, h, w)))
		layer_0 = torch.cat((layer_1, layer_0), 1)

		# Output
		output = self.conv_final(layer_0)

		return output


	def imgs_to_patches(self, device, imgs, n_patches, patch_size):
		h, w, d = imgs.shape

		patches = torch.zeros(n_patches**2, d, h * w // n_patches**2)

		for idx in range(d):
			img = imgs[:,:,idx]

			for i in range(n_patches):
				for j in range(n_patches):
					patch = img[i * patch_size : (i+1) * patch_size, j * patch_size : (j+1) * patch_size]
					patches[i * n_patches + j, idx] = patch.flatten()

		return patches.to(device=device, dtype=torch.float)


	def patches_to_imgs(self, patches, n_patches):
		imgs = patches.view(patches.size(1), patches.size(2), n_patches, n_patches)

		return imgs