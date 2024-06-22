import glob
import configparser
import numpy as np
import random
import nibabel as nib
import torch


class DataReader():
	def __init__(self):
		super(DataReader, self).__init__()
		# Parse all arguments
		config = configparser.ConfigParser()
		config.read('config.ini')
		self.config = config

		# Image
		self.width = int(config['Image']['width'])
		self.height = int(config['Image']['height'])

		# Model
		self.patch_size = int(config['Model']['patch_size'])
		self.d_model = int(config['Model']['d_model'])
		self.n_head = int(config['Model']['n_head'])
		self.num_layers = int(config['Model']['num_layers'])
		self.num_classes = int(config['Model']['num_classes'])

		# Training process
		training_set = eval(config['Training']['training_set'])
		testing_set = eval(config['Training']['testing_set'])
		self.epochs = int(config['Training']['epochs'])
		self.batch_size = int(config['Training']['batch_size'])

		# Split training and testing folders
		folders = {'train': training_set, 'test': testing_set}
		self.folders = {'train': [], 'test': []}

		for mode in ['train', 'test']:
			for folder in folders[mode]:
				img_paths = glob.glob(f'data/{folder}/{mode}/images/*')

				for img_path in img_paths:
					patient_id = img_path.split('/')[-1]
					mask_path = f'data/{folder}/{mode}/masks/{patient_id}'
				
					self.folders[mode].append((img_path, mask_path))

		# Shuffle for random order
		random.shuffle(self.folders['train'])
		random.shuffle(self.folders['test'])



	def read_in_batch(self, img_path, mask_path, device):
		# Read img and normalize to 0~1
		img_nib_file = nib.load(img_path)
		img = img_nib_file.get_fdata()
		img = img - np.min(img)
		img = img / np.max(img)

		img = torch.from_numpy(img)
		img = img.to(device=device, dtype=torch.float)

		# Read mask and change to integer
		mask_nib_file = nib.load(mask_path)
		mask = mask_nib_file.get_fdata()
		mask = np.round(mask)
		mask = mask.astype(int)

		mask = torch.from_numpy(mask)
		mask = mask.type(torch.cuda.LongTensor).to(device)
		mask = torch.moveaxis(mask, 2, 0)

		return img, mask