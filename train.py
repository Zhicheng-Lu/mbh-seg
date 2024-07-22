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
	optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
	scaler = torch.cuda.amp.GradScaler(enabled=amp)

	all_scores = []

	os.mkdir(f'checkpoints/model_{time}')

	for epoch in range(data_reader.epochs):
		optimizer.zero_grad(set_to_none=True)
		torch.cuda.empty_cache()

		train_set = data_reader.folders['train']
		test_set = data_reader.folders['test']

		model.train()

		# Train
		for iteration, (imgs_path, masks_path) in enumerate(train_set):
			imgs, masks = data_reader.read_in_batch(imgs_path, masks_path, device)

			# Compute prediction mask and calculate loss value
			with torch.cuda.amp.autocast():
				pred = model(device, imgs)
				entropy_loss = entropy_loss_fn(pred, masks)
				dice_loss = dice_loss_fn(pred, masks)
				loss = entropy_loss + dice_loss

			# format_losses = [str(f' {dice_loss:.5f} ') for dice_loss in dice_losses]
			# format_losses = ''.join(format_losses)

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
				torch.cuda.empty_cache()

		torch.save(model.state_dict(), f'checkpoints/model_{time}/epoch_{str(epoch+1).zfill(3)}.pt')


		# model.eval()

		# Test on train set
		size = len(train_set)
		dice_total = 0.0
		sensitivity_total = 0.0
		specificity_total = 0.0
		overall_total = 0.0
		for iteration, (imgs_path, masks_path) in enumerate(train_set):
			imgs, masks = data_reader.read_in_batch(imgs_path, masks_path, device)

			with torch.no_grad():
				pred = model(device, imgs)
				dice, sensitivity, specificity, overall = dice_loss_fn(pred, masks, testing=True)
				dice_total += dice
				sensitivity_total += sensitivity
				specificity_total += specificity
				overall_total += overall
				# train_scores.append(scores)

		format_train_scores = f'{str(dice_total/size)} {str(sensitivity_total/size)} {str(specificity_total/size)} {str(overall_total/size)}'
		# train_scores = np.array(train_scores)
		# train_scores = np.mean(train_scores, axis=0)
		# format_train_scores = [str(f' {score:.5f} ') for score in train_scores]
		# format_train_scores = ''.join(format_train_scores)


		# Test on test set
		size = len(test_set)
		dice_total = 0.0
		sensitivity_total = 0.0
		specificity_total = 0.0
		overall_total = 0.0
		for iteration, (imgs_path, masks_path) in enumerate(test_set):
			imgs, masks = data_reader.read_in_batch(imgs_path, masks_path, device)

			with torch.no_grad():
				pred = model(device, imgs)
				dice, sensitivity, specificity, overall = dice_loss_fn(pred, masks, testing=True)
				dice_total += dice
				sensitivity_total += sensitivity
				specificity_total += specificity
				overall_total += overall
				# test_scores.append(scores)

		format_test_scores = f'{str(dice_total/size)} {str(sensitivity_total/size)} {str(specificity_total/size)} {str(overall_total/size)}'

		# test_scores = np.array(test_scores)
		# test_scores = np.mean(test_scores, axis=0)
		# format_test_scores = [str(f' {scores:.5f} ') for scores in test_scores]
		# format_test_scores = ''.join(format_test_scores)

		all_scores.append((epoch+1, format_train_scores, format_test_scores))

		# Print to console
		[print(f'\tEpoch {epoch_scores[0]}: train {epoch_scores[1]}, test {epoch_scores[2]}.') for epoch_scores in all_scores]

		# Print to file
		f = open(f'checkpoints/model_{time}/training.txt', 'a')
		[f.write(f'\tEpoch {epoch_scores[0]}: train {epoch_scores[1]}, test {epoch_scores[2]}.\n') for epoch_scores in all_scores]
		f.close()


if __name__ == "__main__":
	main()