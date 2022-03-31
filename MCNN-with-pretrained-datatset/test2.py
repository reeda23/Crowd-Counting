import os
import torch
import numpy as np

from crowd_count import CrowdCounter
import network
#from data_loader import ImageDataLoader
import utils

from my_dataloader import CrowdDataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path =  './data/original/shanghaitech/part_B_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'
#model_path = './final_models/mcnn_shtechB_110.h5'
model_path = 'mcnn_shtechA_660.h5'

output_dir = 'output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)


net = CrowdCounter()

trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0


img_root = 'E:\\GIKI\\1st Semester\\Robotic Vision\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\part_A\\test_data\\images'
gt_dmap_root = 'E:\\GIKI\\1st Semester\\Robotic Vision\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\part_A\\test_data\\ground-truth'
#load test data
#data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)
dataset = CrowdDataset(img_root, gt_dmap_root, 4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#for blob in data_loader:
with torch.no_grad():
	for i, (im_data, gt_data) in enumerate(dataloader):
		#im_data = blob['data']
		#gt_data = blob['gt_density']
		density_map = net(im_data, gt_data)
		density_map = density_map.data.cpu().numpy()
		gt_count = np.sum(gt_data)
		et_count = np.sum(density_map)
		mae += abs(gt_count-et_count)
		mse += ((gt_count-et_count)*(gt_count-et_count))
		if vis:
			utils.display_results(im_data, gt_data, density_map)
		#if save_output:
			#utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

mae = mae/len(dataloader)
mse = np.sqrt(mse/len(dataloader))
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

f = open(file_results, 'w')
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()
