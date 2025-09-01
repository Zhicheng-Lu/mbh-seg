import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import UNet3D
from utils import DiceCELoss


epochs = 50


def main():
	x_paths = glob.glob('data/data/images/*')
	training_x_paths = x_paths[:152] 
	validation_x_paths = x_paths[152:]

	# Model and parameters
	model = UNet3D().to('cuda')
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	criterion = DiceCELoss(dice_weight=0.7, ce_weight=0.3)

	stdout_file = open('checkpoints/train.txt', 'a')

	for epoch in range(epochs):
		# ----- Training -----
		model.train()
		total_train_loss = 0.0

		for i,training_x_path in enumerate(training_x_paths):
			file_name = training_x_path.split('/')[-1]
			training_y_path = f'data/data/masks/{file_name}'

			x = torch.from_numpy(np.load(training_x_path)).to(dtype=torch.float32, device='cuda')
			y = torch.from_numpy(np.load(training_y_path)).to(dtype=torch.float32, device='cuda')

			# for j in range(4):
			# 	row = j // 2
			# 	col = j % 2

			# 	x_patch = x[:,:,:,row*256:(row+1)*256,col*256:(col+1)*256]
			# 	y_patch = y[:,:,:,row*256:(row+1)*256,col*256:(col+1)*256]

			# 	optimizer.zero_grad()
			# 	pred = model(x_patch)

			# 	loss = criterion(pred, y_patch)
			# 	print(f'Epoch {epoch+1}, iteration {i+1}: {loss.item()}')
			# 	# stdout_file.write(f'Epoch {epoch+1}, iteration {i+1}: {loss.item()}\n')
			# 	loss.backward()
			# 	optimizer.step()

			# 	total_train_loss += loss.item()
			optimizer.zero_grad()
			pred = model(x)
			loss = criterion(pred, y)
			print(f'Epoch {epoch+1}, iteration {i+1}: {loss.item()}')
			loss.backward()
			optimizer.step()

			total_train_loss += loss.item()


		avg_train_loss = total_train_loss / (len(training_x_paths) * 1)
		torch.save(model.state_dict(), f'checkpoints/{str(epoch+1).zfill(3)}.pt')


		# ----- Validation -----
		model.eval()
		total_val_loss = 0.0

		with torch.no_grad():
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
				# 	loss = criterion(pred, y_patch)
				# 	total_val_loss += loss.item()

				pred = model(x)
				loss = criterion(pred, y)
				total_val_loss += loss.item()

		avg_val_loss = total_val_loss / (len(validation_x_paths) * 1)

		print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
		stdout_file.write(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n')

	stdout_file.close()




if __name__ == '__main__':
	main()