import torch
from torch import nn


# Dice Loss Function
class Diceloss(torch.nn.Module):
	def __init__(self):
		super(Diceloss, self).__init__()

	def forward(self, pred, mask):
		num_classes = pred.size(1)
		pred_softmax = nn.functional.softmax(pred, dim=1).float()

		total_loss = 0.0

		for i in range(1, num_classes):
			class_mask = mask * (mask == i)
			class_mask = class_mask / i
			pred_class_mask = pred_softmax[:,i,:,:]
			overlap = class_mask * pred_class_mask
			area_pred = torch.sum(pred_class_mask)
			area_mask = torch.sum(class_mask)
			area_overlap = torch.sum(overlap)

			loss = 1 - (2 * area_overlap + 1) / (area_pred + area_mask + 1)
			total_loss += loss

		return total_loss