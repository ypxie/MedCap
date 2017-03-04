from __future__ import print_function
import argparse

import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn

from resnet import ResNet, BasicBlock, spatial_pool

def start_end(a,b,c):
    return a*c//b, int(math.ceil(float((a+1)*c)/b))

def spatialAdaAvgPool(x, oH, oW):
    B, C, iH, iW  = x.size()
    output = Variable(x.data.new(B, C, oH, oW))
    for oh in xrange(oH):
        for ow in xrange(oW):
            i1, i2 = start_end(oh, oH, iH)

            j1, j2 = start_end(ow, oW, iW)

            output[:, :, oh, ow] = x[:,:,i1:i2,j1:j2].mean(2).mean(3).squeeze(3).squeeze(2)
    return output

def baldder_res1net8(pretrained=False, model_path = None,feature_maps=[32,64,128,256]):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes= 4, feature_maps = feature_maps)
    if pretrained:
        model.load_state_dict(model_path)
    return model

class BladderResnet(nn.Module):
    def __init__(self, num_classes=4,feature_maps=[32,64,128,256], 
                 model_path = None, pretrained=False):
        super(BladderResnet, self).__init__()

        _model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes= num_classes, feature_maps = feature_maps)
        if pretrained:
            _model.load_state_dict(model_path)
        self.resnet = _model
    
    def forward(self, img):
        return self.resnet.forward(img)

    def get_feature(self, x):
        '''
        Parameter:
        ---------
        x: Tensor of shape (Nsample, channel, row, col)
        
        Returns:
        tuple(fc, att_list):
            fc: Tensor of shape (Nsample, channel of last conv)
            att_list: list of Tensor (Nsample, row, col, channel)
        '''
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        #fc = x.mean(2).mean(3).squeeze()
        fc = self.resnet.avgpool(x)
        att = x.permute(0, 2, 3,1)
        #att = spatialAdaAvgPool(x,14,14).squeeze().permute(1, 2, 0)

        return fc, [att]

class SimpleVGG(nn.Module):
    def __init__(self, num_classes=4,feature_maps=[32,64,128,256], 
                 model_path = None, pretrained=False):
        super(BladderResnet, self).__init__()

        _model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes= num_classes, feature_maps = feature_maps)
        if pretrained:
            _model.load_state_dict(model_path)
        self.resnet = _model
    
    def forward(self, img):
        return self.resnet.forward(img)

    def get_feature(self, x):
        '''
        Parameter:
        ---------
        x: Tensor of shape (Nsample, channel, row, col)
        
        Returns:
        tuple(fc, att_list):
            fc: Tensor of shape (Nsample, channel of last conv)
            att_list: list of Tensor (Nsample, row, col, channel)
        '''
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        #fc = x.mean(2).mean(3).squeeze()
        fc = self.resnet.avgpool(x)
        att = x.permute(0, 2, 3,1)
        #att = spatialAdaAvgPool(x,14,14).squeeze().permute(1, 2, 0)

        return fc, [att]
