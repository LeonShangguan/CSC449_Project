import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet
import math
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class net(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=43):
        super().__init__()
        if model_name.lower() == 'resnet34':
            self.backbone = ptcv_get_model("resnet34", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Linear(512, num_classes)
        elif model_name.lower() == 'efficientnet_b7':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b7')

            self.backbone._fc = nn.Sequential(nn.Linear(2560, 256),
                                              Mish(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(256, num_classes))
        elif model_name.lower() == 'se_resnext101':
            self.backbone = ptcv_get_model("seresnext101_32x4d", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Sequential(nn.Linear(2048, 256),
                                                 Mish(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(256, num_classes))
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.backbone(x)

        return x

  
