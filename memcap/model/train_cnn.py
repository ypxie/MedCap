import sys
import os
import argparse

home = os.path.expanduser('~')
projroot = os.path.join('..','..') 

dataroot = os.path.join(home,'DataSet/Bladder_Caption/Augmented/')
trainingimagefolder = os.path.join(dataroot,'Img')
modelroot = os.path.join(projroot, 'Data')

trainingset = 'Bladder'
#modelsubfolder = 'deep_conclusion_bladder'
modelsubfolder = 'residule_conclusion_bladder'

modelfolder = os.path.join(modelroot, 'Model',trainingset,modelsubfolder)

sys.path.insert(0, os.path.join('..', 'proj_utils') )


from time import time
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

from model.res_model import baldder_res_18
from train_eng import get_mean_std

from utils.local_utils import *
from utils.torch_utils import *
from utils.metric import cls_accuracy
from utils.data_augmentor import ImageGenerator

from fuel.cnn_dataloader import CNNData
import json
import h5py

trainingSplitFile = os.path.join(dataroot, 'train_list.json')
validSplitFile    = os.path.join(dataroot, 'valid_list.json')
referDictFile     = os.path.join(dataroot, 'images_caption_refer.json')
h5dataFile        = os.path.join(dataroot, 'images_caption.h5')


with open(trainingSplitFile) as data_file:    
    trainingSplitDict = json.load(data_file)
with open(validSplitFile) as data_file:    
    validSplitDict = json.load(data_file)
with open(referDictFile) as data_file:
    refer_dict = json.load(data_file)

h5_data = h5py.File(h5dataFile,'r')

train_file_list =  trainingSplitDict['img']
valid_file_list =  validSplitDict['img']

split_dict = {'train': train_file_list, 'valid': test_file_list}


if  __name__ == '__main__':
    nb_class = 4

 
    parser = argparse.ArgumentParser(description = 'Bladder Classification')

    parser.add_argument('--reuse_weigths', action='store_false', default=True,
                        help='continue from last checkout point')

    parser.add_argument('--show_progress', action='store_false', default=True,
                        help='show the training process using images')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--maxepoch', type=int, default=128, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for training')
    

    parser.add_argument('--no_cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        strumodel.cuda()
    

    optimizer = optim.SGD(strumodel.parameters(), lr=args.lr, momentum=args.momentum)



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
                                    number_repeat= number_repeat, 
                                    dim_ordering='default')
   

    weightspath = os.path.join(modelfolder,'weights.pth')
    best_weightspath = os.path.join(modelfolder,'best_weights.pth')

    if reuse_weigths == 1 and os.path.isfile(best_weightspath):
        strumodel.load_state_dict(torch.load(best_weightspath) )# 12)

    best_score = 0
    batch_count = 0
    for epochNumber in range(args.maxepoch):
        
        for data_batch, label_batch in train_flow:
            for X_batch, Y_batch in mydata_augmentor(data_batch, label_batch, args.batch_size):
                optimizer.zero_grad()
                pred = strumodel.forward(X_data)
                loss = creteria(pred, label)
                loss.backward()
                optimizer.step()
                batch_count += 1
                assert not np.isnan(np.mean(loss)) ,"nan error"
            
            if np.mod(batch_count, args.valid_freq) == 0:
                batch_count = 0
                acc = validate(model, valid_flow)
                cur_weights = strumodel.state_dict()
    
                print('\nTesting loss: {}, acc: {}, best_score: {}'.format(loss, acc, best_score))
                if acc >=  best_score:
                    best_score = acc
                    print('update to new best_score: {}'.format(best_score))
                    best_weight = strumodel.state_dict()
                    torch.save(best_weight, best_weightspath)
                elif best_score - acc > 0.2 * acc: 
                    strumodel.load_state_dict(best_weight)
                    print('weights have been reset to best_weights!')
                torch.save(cur_weights, weightspath)
        
        cur_weights = strumodel.state_dict()
        torch.save(cur_weights, weightspath)
                
def creteria(pred, label):
    # both of them should be Tensor (N, dim)
    target = to_device(Variable(torch.from_numpy(label.cpu().data.argmax(axis=1))).long(), pre)
    _, target = label.topk(1, dim=1)
    loss = F.nll_loss(F.log_softmax(pred), target)
    return loss

def validate(model, valid_flow):
    acc = []
    pred_list, target_list = [], []
    for data, label in valid_flow:
        pred = model.forward(data)
        target = label.topk(1,dim=1)
        pred_list.append(pred)
        target_list.append(target)
    
    acc = cls_accuracy(torch.cat(pred_list,0), torch.cat(target_list,0))
    return acc
