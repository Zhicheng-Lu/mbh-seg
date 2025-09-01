import torch
import torch.nn as nn


class DiceCELoss(nn.Module):
	def __init__(self, dice_weight=0.7, ce_weight=0.3):
		super().__init__()
		self.dice_weight = dice_weight
		self.ce_weight = ce_weight
		self.ce = nn.CrossEntropyLoss()

	def forward(self, logits, target, sumup=True):
		# logits: (B,6,D,H,W), target: (B,6,D,H,W)
		
		# Cross entropy (multi-class soft)
		# log_probs = F.log_softmax(logits, dim=1)
		# ce = -(target * log_probs).sum(dim=1).mean()
		target_long = target.argmax(dim=1)
		ce = self.ce(logits, target_long)


		# Dice
		probs = torch.softmax(logits, dim=1)
		num = 2 * (probs * target).sum(dim=(0,2,3,4))
		# den = (probs.pow(2) + target.pow(2)).sum(dim=(0,2,3,4)) + 1e-6
		den = (probs + target).sum(dim=(0,2,3,4)) + 1e-6

		dice_per_class = num / den

		# Mask: ignore classes that are absent in target
		mask = torch.sum(target, dim=(0,2,3,4)) > 0  # (C,)

		if not sumup:
			return dice_per_class, mask

		dice = 1 - dice_per_class[1:].sum() / mask[1:].sum().clamp(min=1)

		# dice = 1 - (num / den).mean()

		return self.ce_weight * ce + self.dice_weight * dice