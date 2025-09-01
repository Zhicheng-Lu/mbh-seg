import os
import glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import UNet3D
from utils import DiceCELoss


def main():
	x_paths = glob.glob('data/data/images/*')
	validation_x_paths = x_paths[152:]

	# Model and parameters
	model = UNet3D().to('cuda')
	model.load_state_dict(torch.load("checkpoints/044.pt"))
	criterion = DiceCELoss(dice_weight=0.7, ce_weight=0.3)
	model.eval()

	with torch.no_grad():
		loss_by_class_list = []
		class_present_list = []

		for i,validation_x_path in enumerate(validation_x_paths):
			file_name = validation_x_path.split('/')[-1]
			validation_y_path = f'data/data/masks/{file_name}'

			x = torch.from_numpy(np.load(validation_x_path)).to(dtype=torch.float32, device='cuda')
			y = torch.from_numpy(np.load(validation_y_path)).to(dtype=torch.float32, device='cuda')

			# for j in range(4):
			# 	row = j // 2
			# 	col = j % 2

			# 	x_patch = x[:,:,:,row*256:(row+1)*256,col*256:(col+1)*256]
			# 	y_patch = y[:,:,:,row*256:(row+1)*256,col*256:(col+1)*256]

			# 	pred = model(x_patch)
			# 	loss = criterion(pred, y_patch, sumup=False)
			# 	print(loss)
			pred = model(x)
			loss = criterion(pred, y, sumup=False)
			loss_by_class = loss[0].detach().to('cpu').numpy()
			class_present = loss[1].detach().to('cpu').numpy()
			class_present = class_present.astype(int)

			loss_by_class_list.append(loss_by_class)
			class_present_list.append(class_present)

	loss_by_class_list = np.array(loss_by_class_list)
	class_present_list = np.array(class_present_list)

	loss_by_class_list = np.sum(loss_by_class_list, axis=0)
	class_present_list = np.sum(class_present_list, axis=0)

	print(loss_by_class_list, class_present_list, loss_by_class_list / class_present_list)


if __name__ == '__main__':
	main()