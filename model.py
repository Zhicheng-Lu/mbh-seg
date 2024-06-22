from torch import nn
import torch
from utils.PositionalEncoding import PositionalEncoding


class Segmentation(nn.Module):
	def __init__(self, data_reader):
		super(Segmentation, self).__init__()
		self.data_reader = data_reader
		self.patch_size = data_reader.patch_size
		self.n_patches = data_reader.height // self.patch_size
		d_model = data_reader.d_model
		n_head = data_reader.n_head
		num_layers = data_reader.num_layers
		num_classes = data_reader.num_classes

		self.patchify = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
		self.patch_embedding = nn.Sequential(
			nn.Linear(in_features=self.patch_size*self.patch_size, out_features=d_model)
		)

		self.positional_encoding = PositionalEncoding(d_model)

		self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

		self.output_projection = nn.Sequential(
			nn.Linear(in_features=d_model, out_features=self.patch_size*self.patch_size*num_classes)
		)


	def forward(self, device, img): # img -> h * w * d
		self.num_slices = img.shape[2]
		img = self.img_to_patches(device, img, self.n_patches, self.patch_size) # img -> seq * 1 * (p_size*p_size)
		img = img.view(img.size(0)*img.size(1), 1, img.size(2))
		img = self.patch_embedding(img) # img -> seq * 1 * d_model
		img = self.positional_encoding(img) # img -> seq * 1 * d_model
		img = self.transformer_encoder(img) # img -> seq * 1 * d_model
		img = self.output_projection(img) # img -> seq * 1 * (p_size*p_size*n_class)
		img = self.patches_to_img(img) # img -> d * n_class * h * w

		return img


	def img_to_patches(self, device, volume, n_patches, patch_size):
		h, w, d = volume.shape

		patches = torch.zeros(d, n_patches**2, h * w // n_patches**2)

		for idx in range(volume.size(2)):
			img = volume[:,:,idx]

			for i in range(n_patches):
				for j in range(n_patches):
					patch = img[i * patch_size : (i+1) * patch_size, j * patch_size : (j+1) * patch_size]
					patches[idx, i * n_patches + j] = patch.flatten()

		return patches.to(device=device, dtype=torch.float)


	def patches_to_img(self, img):
		width = self.data_reader.width
		height = self.data_reader.height
		patch_size = self.data_reader.patch_size
		num_classes = self.data_reader.num_classes
		img = img.view(int(height/patch_size), int(width/patch_size), self.num_slices, patch_size, patch_size, num_classes)
		img = img.view(-1, img.size(1), img.size(2), img.size(4), img.size(5))
		img = img.view(img.size(2), img.size(4), img.size(0), -1)

		return img