# ====================================================>>
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/9/20                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================>>


import torch.nn as nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid())

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        token = self.avg_pool(x).view(batch_size, channels)
        token = self.fc(token).view(batch_size, channels, 1, 1)
        return x * token
