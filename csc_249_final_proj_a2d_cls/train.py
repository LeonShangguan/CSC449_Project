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
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import net
import time
from utils.eval_metrics import Precision, Recall, F1
import torchvision
import torch.optim as optim
from apex import amp
import warnings
from torchcontrib.optim import SWA
warnings.filterwarnings('ignore')

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12) # you can make changes

    # Define model, Loss, and optimizer
    model = net('se_resnext101')
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'net_F.ckpt')))
    # for i, param in model.backbone.features.named_parameters():
    #     if 'stage4' or 'stage3' or 'stage2' or 'stage1'in i:
    #         print(i)
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()

    # print(list(filter(lambda p: p.requires_grad, model.parameters())))    
    base_optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # base_optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.005, momentum=0.9)
    optimizer = SWA(base_optimizer, swa_start=1, swa_freq=5, swa_lr=0.005)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.00005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=5e-5, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 20, 40, 75], gamma=0.25)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    # Train the models
    total_step = len(data_loader)

    best_P, best_R, best_F = 0, 0, 0
    for epoch in range(args.num_epochs):
        # scheduler.step()
        print('epoch:{}, lr:{}'.format(epoch, scheduler.get_lr()[0]))

        t1 = time.time()
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            # Forward, backward and optimize
            outputs = model(images)
            loss = criterion(outputs, labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (i+1) % 1 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)

                optimizer.step()
                optimizer.zero_grad()

        # optimizer.swap_swa_sgd()
                # scheduler.step()

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))
        optimizer.swap_swa_sgd()
            # Save the model checkpoints
            # if (i + 1) % args.save_step == 0:
            #     torch.save(model.state_dict(), os.path.join(
            #         args.model_path, 'net.ckpt'))
        scheduler.step()
        t2 = time.time()
        print('Time Spend per epoch: ', t2 - t1)

        if epoch > -1:
            val_dataset = a2d_dataset.A2DDataset(val_cfg, args.dataset_path)
            val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

            X = np.zeros((val_data_loader.__len__(), 43))
            Y = np.zeros((val_data_loader.__len__(), 43))
            model.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(val_data_loader):
                    # mini-batch
                    images = data[0].to(device)
                    labels = data[1].type(torch.FloatTensor).to(device)
                    # output = model(images).cpu().detach().numpy()
                    output = model(images)
                    output = (torch.nn.functional.sigmoid(output)).cpu().detach().numpy()
                    target = labels.cpu().detach().numpy()
                    output[output >= 0.5] = 1
                    output[output < 0.5] = 0
                    X[batch_idx, :] = output
                    Y[batch_idx, :] = target

            P = Precision(X, Y)
            R = Recall(X, Y)
            F = F1(X, Y)
            print('Precision: {:.1f} Recall: {:.1f} F1: {:.1f}'.format(100 * P, 100 * R, 100 * F))

            if(P>best_P):
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net_P.ckpt'))
                best_P = P
            if (R>best_R):
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net_R.ckpt'))
                best_R = R
            if (F>best_F):
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'net_F.ckpt'))
                best_F = F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()
    print(args)
main(args)
