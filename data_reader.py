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
		self.f_size = int(config['Model']['f_size'])
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
				imgs_paths = glob.glob(f'data/{folder}/{mode}/images/*')

				for imgs_path in imgs_paths:
					patient_id = imgs_path.split('/')[-1]
					masks_path = f'data/{folder}/{mode}/masks/{patient_id}'
				
					self.folders[mode].append((imgs_path, masks_path))

		# Shuffle for random order
		random.shuffle(self.folders['train'])
		random.shuffle(self.folders['test'])



	def read_in_batch(self, imgs_path, masks_path, device, to_torch=True):
		# Read imgs and normalize to 0~1
		imgs_nib_file = nib.load(imgs_path)
		imgs = imgs_nib_file.get_fdata()
		if to_torch:
			imgs = imgs - np.min(imgs)
			imgs = imgs / np.max(imgs)

			imgs = torch.from_numpy(imgs)
			imgs = imgs.to(device=device, dtype=torch.float)

		# Read masks and change to integer
		masks_nib_file = nib.load(masks_path)
		masks = masks_nib_file.get_fdata()
		masks = np.round(masks)
		masks = masks.astype(int)

		if to_torch:
			masks = torch.from_numpy(masks)
			masks = masks.type(torch.cuda.LongTensor).to(device)
			masks = torch.moveaxis(masks, 2, 0)

		return imgs, masks