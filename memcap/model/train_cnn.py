import sys
import os
import argparse

home = os.path.expanduser('~')
#dirname = os.path.dirname(__file__)
#print dirname

projroot = os.path.join(os.getcwd()) 

dataroot = os.path.join(projroot,'Data', 'bladder_data')
trainingimagefolder = os.path.join(dataroot,'Img')
modelroot = os.path.join(projroot, 'Data')
trainingset = 'Bladder'
modelsubfolder = 'residule_conclusion_bladder'
modelfolder = os.path.join(modelroot, 'Model',trainingset,modelsubfolder)
sys.path.insert(0, os.path.join(projroot, 'memcap') )

from time import time
import numpy as np

from model.res_model import BladderResnet

from proj_utils.local_utils import *
from proj_utils.torch_utils import *
from proj_utils.data_augmentor import ImageDataGenerator

from fuel.cnn_dataloader import CNNData
import json
import h5py

import torch.optim as optim

trainingSplitFile = os.path.join(dataroot, 'train_list.json')
validSplitFile    = os.path.join(dataroot, 'valid_list.json')
referDictFile     = os.path.join(dataroot, 'images_caption_ref.json')
h5dataFile        = os.path.join(dataroot, 'images_caption.h5')


with open(trainingSplitFile) as data_file:    
    trainingSplitDict = json.load(data_file)
with open(validSplitFile) as data_file:    
    validSplitDict = json.load(data_file)
with open(referDictFile) as data_file:
    refer_dict = json.load(data_file)

#h5_data = h5py.File(h5dataFile,'r')
train_file_list =  trainingSplitDict['img']
valid_file_list =  validSplitDict['img']

split_dict = {'train': train_file_list, 'valid': valid_file_list}

if  __name__ == '__main__':
    nb_class = 4
    parser = argparse.ArgumentParser(description = 'Bladder Classification')

    parser.add_argument('--reuse_weigths', action='store_false', default=True,
                        help='continue from last checkout point')

    parser.add_argument('--show_progress', action='store_false', default=True,
                        help='show the training process using images')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--maxepoch', type=int, default=128, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for training')
    
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--valid_freq', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    strumodel = BladderResnet()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        strumodel.cuda()
    

    optimizer = optim.SGD(strumodel.parameters(), lr=args.lr, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)


    with h5py.File(h5dataFile,'r') as h5_data:
        data_loader = CNNData(h5_data, batch_size=  args.batch_size, 
                            split_dict = split_dict, refer_dict = refer_dict)
        train_flow = data_loader.get_flow('train')
        valid_flow = data_loader.get_flow('valid')

        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)
        mydata_augmentor = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        rotation_range= 270,
                                        width_shift_range= 0.2,
                                        height_shift_range= 0.2,
                                        shear_range=0.,
                                        zoom_range=0.,
                                        channel_shift_range=3.,
                                        fill_mode='reflect',
                                        cval=0.,
                                        horizontal_flip=True,
                                        vertical_flip= True,
                                        rescale=None,
                                        preprocessing_function=None,
                                        elastic = False,
                                        elastic_label = False,
                                        transform_label = False,
                                        number_repeat= 1, 
                                        dim_ordering='default')
    

        weightspath = os.path.join(modelfolder,'weights.pth')
        best_weightspath = os.path.join(modelfolder,'best_weights.pth')

        if args.reuse_weigths == 1 and os.path.isfile(best_weightspath):
            strumodel.load_state_dict(torch.load(best_weightspath) )# 12)

        best_score = 0
        batch_count = 0
        for epochNumber in range(args.maxepoch):
            
            for data_batch, label_batch, in train_flow:
                for train_data, train_label in mydata_augmentor.flow(data_batch, label_batch, args.batch_size):
                    optimizer.zero_grad()
                    pred = strumodel.forward(to_variable(train_data, cuda=args.cuda))
                    loss = creteria(pred, to_variable(train_label, cuda=args.cuda) )
                    loss.backward()
                    optimizer.step()
                    batch_count += 1
                    assert not np.isnan(np.mean(loss.data.cpu().numpy())) ,"nan error"
                    print('batch count: {}'.format(batch_count))
                    if np.mod(batch_count, args.valid_freq) == 0:
                        batch_count = 0
                        acc = validate(strumodel, valid_flow, cuda=args.cuda)
                        cur_weights = strumodel.state_dict()
            
                        print('\nTesting loss: {}, acc: {}, best_score: {}'.format(loss, acc, best_score))
                        if acc >=  best_score:
                            best_score = acc
                            print('update to new best_score: {}'.format(best_score))
                            best_weight = strumodel.state_dict()
                            torch.save(best_weight, best_weightspath)
                        elif best_score - acc > 3 * acc: 
                            strumodel.load_state_dict(best_weight)
                            print('weights have been reset to best_weights!')
                        torch.save(cur_weights, weightspath)
            
            cur_weights = strumodel.state_dict()
            torch.save(cur_weights, weightspath)
                    

