from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from network import net
import time
from utils.eval_metrics import Precision, Recall, F1

import torchvision

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # define load your model here
    model = net('se_resnext101')
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'seresnext101_val_55.1_56.2.ckpt')))
    
    X = np.zeros((data_loader.__len__(), args.num_cls))
    Y = np.zeros((data_loader.__len__(), args.num_cls))
    print(data_loader.__len__())
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            # output = model(images).cpu().detach().numpy()
            output = model(images)
            output = (torch.nn.functional.sigmoid(output)).cpu().detach().numpy()
            target = labels.cpu().detach().numpy()
            output[output >= 0.4] = 1
            output[output < 0.4] = 0
            X[batch_idx, :] = output
            Y[batch_idx, :] = target
        
    P = Precision(X, Y)
    R = Recall(X, Y)
    F = F1(X, Y)
    print('Precision: {:.1f} Recall: {:.1f} F1: {:.1f}'.format(100 * P, 100 * R, 100 * F))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    args = parser.parse_args()

main(args)
