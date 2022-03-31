# %%
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM

from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import numpy as np
import cv2

def cal_mae(img_root, gt_dmap_root, model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device = torch.device("cuda")
    mcnn = MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)
    mcnn.eval()
    mae = 0
    mse = 0
    with torch.no_grad():
        for i, (img, gt_dmap) in enumerate(dataloader):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = mcnn(img)
            mae += abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            gt_count = gt_dmap.data.sum()
            et_count = et_dmap.data.sum()
            mse += ((gt_count-et_count)*(gt_count-et_count))
            del img, gt_dmap, et_dmap

    print("model_param_path:"+model_param_path +
          " MAE:"+str(mae/len(dataloader)))
    mse = np.sqrt(mse.cpu().numpy()/len(dataloader))
    print("MSE:"+ str(mse))


def estimate_density_map(img_root, gt_dmap_root, model_param_path, index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda")
    mcnn = MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)
    mcnn.eval()
    for i, (img, gt_dmap) in enumerate(dataloader):
        if i == index:
            im = img
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = mcnn(img).detach()
            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            density_map = 255*et_dmap/np.max(et_dmap)
            print(et_dmap.shape)
            #plt.imshow(im.reshape(680, 1024, 3))
            plt.imshow(et_dmap, cmap='inferno')
            plt.waitforbuttonpress()
            break


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    img_root = '/home/faizi/Downloads/Compressed/MCNN-pytorch-master/ShanghaiTech/part_A/test_data/images'
    gt_dmap_root = '/home/faizi/Downloads/Compressed/MCNN-pytorch-master/ShanghaiTech/part_A/test_data/ground-truth'
    model_param_path = '/home/faizi/Downloads/Compressed/MCNN-pytorch-master/checkpoints/epoch_1999.param'
    #cal_mae(img_root, gt_dmap_root, model_param_path)
    estimate_density_map(img_root, gt_dmap_root, model_param_path, 0)
