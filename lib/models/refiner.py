from __future__ import absolute_import
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


# RefineNet
class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, bias=True, bn=True, leaky=True):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.bn = bn
        self.leaky = leaky

        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p_dropout)

        self.dense1 = nn.Linear(self.l_size, self.l_size, bias=bias)
        self.dense2 = nn.Linear(self.l_size, self.l_size, bias=bias)
        self.dense3 = nn.Linear(self.l_size, self.l_size, bias=bias)
        self.dense4 = nn.Linear(self.l_size, self.l_size, bias=bias)

        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(self.l_size)
            self.batch_norm2 = nn.BatchNorm1d(self.l_size)
            self.batch_norm3 = nn.BatchNorm1d(self.l_size)
            self.batch_norm4 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.dense1(x)
        if self.bn:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.dense2(y)
        if self.bn:
            y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        y = self.dense3(out)
        if self.bn:
            y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.dense4(y)
        if self.bn:
            y = self.batch_norm4(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = out + y

        return out

class RefineNet(nn.Module):
    def __init__(self,
                 cfg,
                 dim=3,
                 linear_size=2048,
                 num_stage=2,
                 p_dropout=0.5,
                 bias=True,
                 bn=True,
                 leaky=True):
        super(RefineNet, self).__init__()

        self.linear_size = linear_size
        self.bn = bn
        self.leaky = leaky
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.dim = dim

        # 2D joints
        self.input_size = (cfg.MODEL.NUM_JOINTS - 1) * self.dim
        # 3D joints
        self.output_size = (cfg.MODEL.NUM_JOINTS - 1) * self.dim

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout, bias=bias, bn=self.bn,
                                             leaky=self.leaky))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.dense1 = nn.Linear(self.input_size, self.linear_size, bias=bias)
        self.dense2 = nn.Linear(self.linear_size, self.output_size, bias=bias)
        self.dense3 = nn.Linear(self.output_size, self.linear_size, bias=bias)
        self.dense4 = nn.Linear(self.linear_size, self.output_size, bias=bias)

        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
            self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        self.dropout = nn.Dropout(self.p_dropout)


    def forward(self, x):
        y = self.dense1(x)
        if self.bn:
            y = self.batch_norm1(y)
        y = self.relu(y)
        inp = self.dropout(y)

        s1 = self.linear_stages[0](inp)

        p1 = self.dense2(s1)

        y = self.dense3(p1)
        if self.bn:
            y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = s1 + y + inp

        y = self.linear_stages[1](y)

        y = inp + y

        p2 = self.dense4(y)

        return p1, p2 #p1 is the original model, p2 is further refined


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        # nn.init.constant_(m.weight, 0.)
        # nn.init.constant_(m.bias, 0.)

def get_refiner (cfg, is_train, weights, **kwargs):
    model = RefineNet(cfg, **kwargs)
    
    if weights:
        model.load_state_dict(torch.load(weights)['state_dict'])
    if not is_train:
        model.apply(weight_init)
    
    return model

#if __name__ == 'main':
    #print(output[0].shape)
