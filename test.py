import os
import glob
import numpy as np
from datetime import datetime
from data_reader import DataReader
import torch
from torch import nn
from model import Segmentation
from utils.loss import Diceloss
import nibabel as nib
import cv2


def testing_full():
	data_reader = DataReader()

	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")
	time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	# Define loss and model
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader)
	model.load_state_dict(torch.load("checkpoints/model.pt"))
	model = model.to(device)
	model.eval()

	losses = []

	# Create output folder
	os.mkdir(f'test/test_full_{time}')

	test_set = data_reader.folders['test']
	for iteration, (imgs_path, masks_path) in enumerate(test_set):
		patient_id = (imgs_path.split('/')[-1]).split('.')[0]
		os.mkdir(f'test/test_full_{time}/{patient_id}')

		imgs, masks = data_reader.read_in_batch(imgs_path, masks_path, device)
		with torch.no_grad():
			pred = model(device, imgs)
			print(entropy_loss_fn(pred, masks), dice_loss_fn(pred, masks))
			
		pred_masks = pred.detach().cpu().numpy()
		pred_masks = np.argmax(pred_masks, axis=1)

		# Get numpy images and masks
		imgs, masks = data_reader.read_in_batch(imgs_path, masks_path, device, False)

		# Map to different colors
		color_map = np.array([[0,0,0], [197,70,70], [73,72,199], [170,43,170], [168,181,115], [9,134,134]])
		color_masks = color_map[masks]
		color_pred_masks = color_map[pred_masks]
		
		for i, (img, mask, pred_mask) in enumerate(zip(np.moveaxis(imgs, 2, 0), np.moveaxis(color_masks, 2, 0), color_pred_masks)):
			cv2.imwrite(f'test/test_full_{time}/{patient_id}/{i+1}_input.jpg', img)
			cv2.imwrite(f'test/test_full_{time}/{patient_id}/{i+1}_mask.jpg', mask)
			cv2.imwrite(f'test/test_full_{time}/{patient_id}/{i+1}_output.jpg', pred_mask)

		return


def transform(x):
	return np.array([0, 0, 0])


if __name__ == "__main__":
	testing_full()