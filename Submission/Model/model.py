import torch
import torch.nn as nn

class DoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(DoubleConv, self).__init__()
		self.net = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
			# nn.BatchNorm3d(out_ch),
			# nn.InstanceNorm3d(out_ch, affine=True, track_running_stats=False),
			nn.ReLU(inplace=True),
			nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
			# nn.BatchNorm3d(out_ch),
			# nn.InstanceNorm3d(out_ch, affine=True, track_running_stats=False),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.net(x)

class UNet3D(nn.Module):
	def __init__(self, in_channels=1, out_channels=6, base_channels=16):
		super(UNet3D, self).__init__()

		self.encoder1 = DoubleConv(in_channels, base_channels)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

		self.encoder2 = DoubleConv(base_channels, base_channels * 2)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

		self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

		self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

		self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
		self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)

		self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
		self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)

		self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
		self.decoder1 = DoubleConv(base_channels * 2, base_channels)

		self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

	def forward(self, x):
		e1 = self.encoder1(x)
		e2 = self.encoder2(self.pool1(e1))
		e3 = self.encoder3(self.pool2(e2))

		b = self.bottleneck(self.pool3(e3))

		d3 = self.up3(b)
		d3 = self.decoder3(torch.cat([d3, e3], dim=1))

		d2 = self.up2(d3)
		d2 = self.decoder2(torch.cat([d2, e2], dim=1))

		d1 = self.up1(d2)
		d1 = self.decoder1(torch.cat([d1, e1], dim=1))

		out = self.out_conv(d1)
		return out