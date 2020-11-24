from loader import a2d_dataset
import argparse
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
# from network import Res152_MLMC
import pickle
import torch.nn as nn
from network import net
import time
import warnings
warnings.filterwarnings('ignore')

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valid_cls = [
    11, 12, 13, 15, 16, 17, 18, 19,
    21, 22, 26, 28, 29,
    34, 35, 36, 39,
    41, 43, 44, 45, 46, 48, 49,
    54, 55, 56, 57, 59,
    61, 63, 65, 66, 67, 68, 69,
    72, 73, 75, 76, 77, 78, 79]
class_names = np.array([
    'background',
    'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
    'none',
    'adult-climbing',
    'adult-crawling',
    'adult-eating',
    'adult-flying',
    'adult-jumping',
    'adult-rolling',
    'adult-running',
    'adult-walking',
    'adult-none',
    'none',
    'baby-climbing',
    'baby-crawling',
    'baby-eating',
    'baby-flying',
    'baby-jumping',
    'baby-rolling',
    'baby-running',
    'baby-walking',
    'baby-none',
    'none',
    'ball-climbing',
    'ball-crawling',
    'ball-eating',
    'ball-flying',
    'ball-jumping',
    'ball-rolling',
    'ball-running',
    'ball-walking',
    'ball-none',
    'none',
    'bird-climbing',
    'bird-crawling',
    'bird-eating',
    'bird-flying',
    'bird-jumping',
    'bird-rolling',
    'bird-running',
    'bird-walking',
    'bird-none',
    'none',
    'car-climbing',
    'car-crawling',
    'car-eating',
    'car-flying',
    'car-jumping',
    'car-rolling',
    'car-running',
    'car-walking',
    'car-none',
    'none',
    'cat-climbing',
    'cat-crawling',
    'cat-eating',
    'cat-flying',
    'cat-jumping',
    'cat-rolling',
    'cat-running',
    'cat-walking',
    'cat-none',
    'none',
    'dog-climbing',
    'dog-crawling',
    'dog-eating',
    'dog-flying',
    'dog-jumping',
    'dog-rolling',
    'dog-running',
    'dog-walking',
    'dog-none',
])

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset_test(test_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # define and load pre-trained model
    model = net('se_resnext101')
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'seresnext_mode_train_56.2_57.8.ckpt')))

    results = np.zeros((data_loader.__len__(), args.num_cls))
    model.eval()

    # prediction and saving
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data.to(device)
            # output = model(images).cpu().detach().numpy()
            output = model(images)
            output = (torch.nn.functional.sigmoid(output)).cpu().detach().numpy()
            output[output >= 0.4] = 1
            output[output < 0.4] = 0
            results[batch_idx, :] = output
    with open('results_zshangg2.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--num_cls', type=int, default=43)

    args = parser.parse_args()

main(args)
