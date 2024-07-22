import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Dice Loss Function
class Diceloss(torch.nn.Module):
	def __init__(self):
		super(Diceloss, self).__init__()

	def forward(self, pred, masks, epsilon=1e-5, testing=False):
		num_classes = pred.size(1)
		masks = F.one_hot(masks, num_classes=num_classes)
		masks = torch.moveaxis(masks, 3, 1)
		pred_softmax = nn.functional.softmax(pred, dim=1).float()

		TP = torch.sum(pred_softmax[:, 1:, :, :] * masks[:, 1:, :, :])
		FP = torch.sum(masks[:, 1:, :, :]) - TP
		FN = torch.sum(pred_softmax[:, 1:, :, :]) - TP
		TN = torch.sum(pred_softmax[:, 0, :, :] * masks[:, 0, :, :])

		dice_loss = 1 - (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)

		if testing:
			dice = 1 - dice_loss
			sensitivity = (TP + epsilon) / (TP + FN + epsilon)
			specificity = (TN + epsilon) / (TN + FP + epsilon)
			overall = 0.5 * dice + 0.25 * sensitivity + 0.25 * specificity

			return dice.item(), sensitivity.item(), specificity.item(), overall.item()

		return dice_loss

		# area_pred = torch.sum(pred_softmax[:, 1:, :, :])
		# area_masks = torch.sum(masks[:, 1:, :, :])
		# area_overlap = torch.sum(pred_softmax[:, 1:, :, :] * masks[:, 1:, :, :])
		
		# loss = 1 - (2 * area_overlap + epsilon) / (area_pred + area_masks + epsilon)

		# if testing:
		# 	area_pred_list = torch.sum(pred_softmax[:, 1:, :, :], dim=(0,2,3))
		# 	area_masks_list = torch.sum(masks[:, 1:, :, :], dim=(0,2,3))
		# 	area_overlap_list = torch.sum(pred_softmax[:, 1:, :, :] * masks[:, 1:, :, :], dim=(0,2,3))

		# 	losses = 1 - (2 * area_overlap_list + epsilon) / (area_pred_list + area_masks_list + epsilon)

		# 	return loss, losses.tolist()

		# return loss