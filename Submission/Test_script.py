import os
import glob
import numpy as np
import torch
import nibabel as nib
from Model.model import UNet3D
from Model.utils import DiceCELoss


def main():
	in_nii_paths = glob.glob('Input/*')
	out_dir = 'Output'

	# Model and parameters
	model = UNet3D().to('cuda')
	model.load_state_dict(torch.load("Model/weights.pt"))
	
	model.eval()

	with torch.no_grad():
		for i,in_nii_path in enumerate(in_nii_paths):
			filename = in_nii_path.split('/')[-1]

			nib_file = nib.load(in_nii_path)
			img = nib_file.get_fdata()
			
			# Clip to [-100,300] and convert to B*1*D*H*W tensor
			img = np.clip(img, -100, 300)
			img = np.moveaxis(img, 2, 0)
			img = img[np.newaxis, np.newaxis, :]
			x = torch.from_numpy(img).to(dtype=torch.float32, device='cuda')
			
			# Predict, convert back to H*W*D
			pred = model(x)
			pred = pred.detach().to('cpu').numpy()
			pred = np.argmax(pred, axis=1)
			pred = np.squeeze(pred)
			pred = np.moveaxis(pred, 0, 2)

			# Add nifty headers to output
			ref_affine = nib_file.affine
			ref_header = nib_file.header.copy()
			gen_nifti = nib.Nifti1Image(pred, affine=ref_affine, header=ref_header)

			# Save output
			nib.save(gen_nifti, f'{out_dir}/{filename}')


if __name__ == '__main__':
	main()