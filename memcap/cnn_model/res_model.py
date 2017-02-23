from __future__ import print_function
import argparse

import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn



class baldder_res_18(nn.Module):
    def __init__(self, num_class, pretrained_model):
        self.pretrained_model = pretrained_model
        self.pretrained_model.fc = nn.Linear(512, num_class)

    
    def forward(self, x):
        return self.pretrained_model(x)

