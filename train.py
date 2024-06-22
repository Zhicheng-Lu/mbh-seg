import os
import glob
import numpy as np
from datetime import datetime
from data_reader import DataReader
import torch
from torch import nn
from torch.cuda import amp
from model import Segmentation
from utils.loss import Diceloss
import nibabel as nib


def main():
	data_reader = DataReader()

	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")
	time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	# Define loss and model
	entropy_loss_fn = nn.CrossEntropyLoss()
	dice_loss_fn = Diceloss()
	model = Segmentation(data_reader)
	model = model.to(device)

	# Define optimier and scaler
	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	losses = []

	os.mkdir(f'checkpoints/model_{time}')

	for epoch in range(data_reader.epochs):
		optimizer.zero_grad(set_to_none=True)
		torch.cuda.empty_cache()

		train_set = data_reader.folders['train']
		test_set = data_reader.folders['test']

		model.train()

		# Train
		for iteration, (img_path, mask_path) in enumerate(train_set):
			img, mask = data_reader.read_in_batch(img_path, mask_path, device)

			# Compute prediction mask and calculate loss value
			with torch.cuda.amp.autocast():
				pred = model(device, img)
				loss = entropy_loss_fn(pred, mask) + dice_loss_fn(pred, mask)

			print(f'Epoch {epoch+1} iteration {iteration} loss: {loss}')
			f = open(f'checkpoints/model_{time}/training.txt', 'a')
			f.write(f'Epoch {epoch+1} iteration {iteration} loss: {loss}\n')
			f.close()

			# Backpropagation
			scaler.scale(loss).backward()

			# Gradient accumulation
			if (iteration+1) % data_reader.batch_size == 0:
				scaler.step(optimizer)
				optimizer.zero_grad(set_to_none=True)
				scaler.update()

		torch.save(model.state_dict(), f'checkpoints/model_{time}/epoch_{str(epoch+1).zfill(3)}.pt')

		model.eval()

		# Test on train set
		train_loss = 0.0
		for iteration, (img_path, mask_path) in enumerate(train_set):
			img, mask = data_reader.read_in_batch(img_path, mask_path, device)

			with torch.no_grad():
				pred = model(device, img)
				loss = entropy_loss_fn(pred, mask) + dice_loss_fn(pred, mask)
				train_loss += loss.item()

		train_loss = train_loss / len(train_set)

		# Test on test set
		test_loss = 0.0
		for iteration, (img_path, mask_path) in enumerate(test_set):
			img, mask = data_reader.read_in_batch(img_path, mask_path, device)

			with torch.no_grad():
				pred = model(device, img)
				loss = entropy_loss_fn(pred, mask) + dice_loss_fn(pred, mask)
				test_loss += loss.item()

		test_loss = test_loss / len(test_set)

		losses.append((epoch+1, train_loss, test_loss))

		# Print to console
		[print(f'\tEpoch {epoch_loss[0]}: train {epoch_loss[1]}, test {epoch_loss[2]}.') for epoch_loss in losses]

		# Print to file
		f = open(f'checkpoints/model_{time}/training.txt', 'a')
		[f.write(f'\tEpoch {epoch_loss[0]}: train {epoch_loss[1]}, test {epoch_loss[2]}.\n') for epoch_loss in losses]
		f.close()


if __name__ == "__main__":
	main()