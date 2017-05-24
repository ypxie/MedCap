import sys
import os
import argparse

home = os.path.expanduser('~')

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
testSplitFile     = os.path.join(dataroot, 'test_list.json')

referDictFile     = os.path.join(dataroot, 'images_caption_ref.json')
h5dataFile        = os.path.join(dataroot, 'images_caption.h5')

with open(trainingSplitFile) as data_file:    
    trainingSplitDict = json.load(data_file)
with open(validSplitFile) as data_file:    
    validSplitDict = json.load(data_file)
with open(testSplitFile) as data_file:    
    testSplitDict = json.load(data_file)
with open(referDictFile) as data_file:
    refer_dict = json.load(data_file)

train_file_list =  trainingSplitDict['img']
valid_file_list =  validSplitDict['img']
test_file_list  =  testSplitDict['img']
#split_dict = {'train': train_file_list, 'valid': valid_file_list, 'test':test_file_list}
split_dict = {'all': train_file_list + valid_file_list + test_file_list}

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

    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

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
        all_flow = data_loader.get_flow('all')

        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)

        weightspath = os.path.join(modelfolder,'weights.pth')
        best_weightspath = os.path.join(modelfolder,'best_weights.pth')
        best_score = 0

        if args.reuse_weigths == 1 and os.path.isfile(best_weightspath):
            best_weights_dict = torch.load(best_weightspath)
            best_score = best_weights_dict.pop('acc_score',0) # because it is not part of the graph
            strumodel.load_state_dict(best_weights_dict)# 12)
            print('reload weights from {}, with score {}'.format(best_weightspath, best_score))
        
        batch_count = 0
        for data_batch, label_batch, in train_flow:
            


                    

