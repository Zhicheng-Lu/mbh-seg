from torch import nn
import torch

class Segmentation(nn.Module):
	def __init__(self, data_reader):
		super(Segmentation, self).__init__()
		self.f_size = data_reader.f_size
		num_classes = data_reader.num_classes

		self.down1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(self.f_size),
			nn.ReLU()
		)
		self.down2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(2*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(2*self.f_size),
			nn.ReLU()
		)
		self.down3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=2*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(4*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(4*self.f_size),
			nn.ReLU()
		)
		self.down4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=4*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(8*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(8*self.f_size),
			nn.ReLU()
		)
		self.down5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(2,2)),
			nn.Conv2d(in_channels=8*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(16*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=16*self.f_size, out_channels=16*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(16*self.f_size),
			nn.ReLU()
			
		)
		self.up1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.BatchNorm2d(8*self.f_size),
			nn.ReLU()
			
		)
		self.up2 = nn.Sequential(
			nn.Conv2d(in_channels=16*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(8*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=8*self.f_size, out_channels=8*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(8*self.f_size),
			nn.ReLU(),
			
			nn.ConvTranspose2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.BatchNorm2d(4*self.f_size),
			nn.ReLU()
			
		)
		self.up3 = nn.Sequential(
			nn.Conv2d(in_channels=8*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(4*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=4*self.f_size, out_channels=4*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(4*self.f_size),
			nn.ReLU(),
			
			nn.ConvTranspose2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.BatchNorm2d(2*self.f_size),
			nn.ReLU()
		)
		self.up4 = nn.Sequential(
			nn.Conv2d(in_channels=4*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(2*self.f_size),
			nn.ReLU(),
			
			nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(2*self.f_size),
			nn.ReLU(),
			
			nn.ConvTranspose2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
			nn.BatchNorm2d(self.f_size),
			nn.ReLU()
			
		)
		self.conv = nn.Sequential(
			nn.Conv3d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.BatchNorm3d(self.f_size),
			nn.ReLU(),
			
			nn.Conv3d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3,3), padding=(1,1,1)),
			nn.BatchNorm3d(self.f_size),
			nn.ReLU(),
			
			nn.Conv3d(in_channels=self.f_size, out_channels=num_classes, kernel_size=(1,1,1))
		)
		# self.aggr_1 = nn.Sequential(
		# 	nn.Conv2d(in_channels=2*self.f_size, out_channels=2*self.f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU()
		# )
		# self.update_1 = nn.Sequential(
		# 	nn.Conv2d(in_channels=4*self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU()
		# )
		# self.aggr_2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU()
		# )
		# self.update_2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=2*self.f_size, out_channels=self.f_size, kernel_size=(3,3), padding=(1,1)),
		# 	nn.ReLU()
		# )
		# self.conv = nn.Sequential(
		# 	nn.Conv2d(in_channels=self.f_size, out_channels=num_classes, kernel_size=(1,1))
		# )
		

	def forward(self, device, cts):
		cts = torch.moveaxis(cts, 2, 0)
		cts = cts[:, None, :, :]

		down1 = self.down1(cts)
		down2 = self.down2(down1)
		down3 = self.down3(down2)
		down4 = self.down4(down3)
		down5 = self.down5(down4)
		up1 = self.up1(down5)
		up1_cat = torch.cat((down4, up1), dim=1)
		up2 = self.up2(up1_cat)
		up2_cat = torch.cat((down3, up2), dim=1)
		up3 = self.up3(up2_cat)
		up3_cat = torch.cat((down2, up3), dim=1)
		up4 = self.up4(up3_cat)
		up4_cat = torch.cat((down1, up4), dim=1)
		up4_cat = torch.moveaxis(up4_cat, 0, 1)
		up4_cat = up4_cat[None, :, :, :, :]
		output = self.conv(up4_cat)
		output = torch.moveaxis(output, 2, 1)
		output = torch.squeeze(output)
		# gnn1 = self.GNN_layer(device, up4_cat, self.aggr_1, self.update_1)
		# gnn2 = self.GNN_layer(device, gnn1, self.aggr_2, self.update_2)
		# output = self.conv(gnn2)

		# print(cts.shape)
		# print(down1.shape)
		# print(down2.shape)
		# print(down3.shape)
		# print(down4.shape)
		# print(down5.shape)
		# print(up1.shape)
		# print(up1_cat.shape)
		# print(up2.shape)
		# print(up2_cat.shape)
		# print(up3.shape)
		# print(up3_cat.shape)
		# print(up4.shape)
		# print(up4_cat.shape)
		# print(conv.shape)
		# print(output.shape)
		# print(up4_cat.shape)
		# print(gnn1.shape)
		# print(gnn2.shape)
		# print(gnn3.shape)

		return output


	def GNN_layer(self, device, in_features, aggr, update):
		features = in_features[:, None, :, :, :]
		features = features.repeat(1,2,1,1,1)


		for n1 in range(len(in_features)):
			similarities = []
			# Find similarities of all pairs
			for n2 in range(len(in_features)):
				if n1 == n2:
					continue
				similarities.append((n2, nn.functional.cosine_similarity(torch.flatten(features[n1,0]), torch.flatten(features[n2,0]), dim=0)))
			# Sort and slice top 5
			similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
			if len(similarities) < 3:
				knn = [similarity[0] for similarity in similarities]
			else:
				knn = [similarities[i][0] for i in range(len(similarities)) if i<3]
			knn = features[knn,0]
			knn = aggr(knn)
			knn = torch.mean(knn, dim=0, keepdim=False)
			features[n1,1] = knn

		features = torch.reshape(features, (features.shape[0], features.shape[1]*features.shape[2], features.shape[3], features.shape[4]))
		features = update(features)


		return features