import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
	def __init__(self, dice_weight=0.5, ce_weight=0.5):
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




def sliding_window_inference(volume, model, patch_size=(256, 256), overlap=0.25, device='cuda'):
	"""
	Args:
		volume: torch.Tensor, shape (1,1,D,512,512)
		model: 3D UNet, outputs (1,6,D,H,W)
		patch_size: tuple (patch_H, patch_W)
		overlap: fraction of overlap between patches (0~1)
		device: 'cuda' or 'cpu'
	Returns:
		output: torch.Tensor, shape (1,6,D,512,512)
	"""
	B, C, D, H, W = volume.shape
	ph, pw = patch_size

	stride_h = int(ph * (1 - overlap))
	stride_w = int(pw * (1 - overlap))

	# Prepare empty output and count map
	output = torch.zeros((B, 6, D, H, W), device=device)
	count_map = torch.zeros((B, 1, D, H, W), device=device)

	volume = volume.to(device)

	# Loop over H and W with sliding window
	for h_start in range(0, H, stride_h):
		for w_start in range(0, W, stride_w):
			h_end = min(h_start + ph, H)
			w_end = min(w_start + pw, W)
			h_start_adj = h_end - ph if h_end - h_start < ph else h_start
			w_start_adj = w_end - pw if w_end - w_start < pw else w_start

			patch = volume[:, :, :, h_start_adj:h_end, w_start_adj:w_end]  # (1,1,D,ph,pw)

			with torch.no_grad():
				pred_patch = model(patch)  # (1,6,D,ph,pw)

			output[:, :, :, h_start_adj:h_end, w_start_adj:w_end] += pred_patch
			count_map[:, :, :, h_start_adj:h_end, w_start_adj:w_end] += 1.0

	# Average overlapping regions
	output = output / count_map
	return output