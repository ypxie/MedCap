import sys
import os
import argparse

home = os.path.expanduser('~')
#dirname = os.path.dirname(__file__)
#print dirname

projroot = os.path.join('..','..')

dataroot = os.path.join(projroot,'Data', 'bladder_data')
trainingimagefolder = os.path.join(dataroot,'Img')
modelroot = os.path.join(projroot, 'Data')
trainingset = 'Bladder'
modelsubfolder = 'reward_model'
modelfolder = os.path.join(modelroot, 'Model',trainingset,modelsubfolder)
sys.path.insert(0, os.path.join(projroot, 'memcap') )

from time import time
import numpy as np
import json
import h5py

from memcap.model.reward_model import RewardModel
from memcap.proj_utils.local_utils import *
from memcap.proj_utils.torch_utils import *
from memcap.fuel.pairGenerator import PairData as Dataloader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

import torch

trainingSplitFile = os.path.join(dataroot, 'train_list.json')
validSplitFile    = os.path.join(dataroot, 'valid_list.json')
referDictFile     = os.path.join(dataroot, 'images_caption_ref.json')
h5dataFile        = os.path.join(dataroot, 'images_caption.h5')
dictionaryFile    = os.path.join(dataroot, 'dictionary.json')

with open(trainingSplitFile) as data_file:    
    trainingSplitDict = json.load(data_file)
with open(validSplitFile) as data_file:    
    validSplitDict = json.load(data_file)
with open(referDictFile) as data_file:
    refer_dict = json.load(data_file)
with open(dictionaryFile) as data_file:
    dictionary = json.load(data_file)['encoder']

train_file_list =  trainingSplitDict['img']
valid_file_list =  validSplitDict['img']

split_dict = {'train': train_file_list}

if  __name__ == '__main__':
    nb_class = 4
    parser = argparse.ArgumentParser(description = 'Bladder Classification')

    parser.add_argument('--reuse_weigths', action='store_false', default=False,
                        help='continue from last checkout point')

    parser.add_argument('--show_progress', action='store_false', default=True,
                        help='show the training process using images')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--maxepoch', type=int, default=12800, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for training')
    
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='the rnn hidden size')
    parser.add_argument('--feature_size', type=int, default=128,
                        help='the final representation vector size before cosine similarity')

    parser.add_argument('--num_layers', type=int, default= True,
                        help='the rnn hidden size')
    parser.add_argument('--brnn', action='store_false', default=False,
                        help='bidirectional rnn')
    parser.add_argument('--showfre', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--savefreq', type=int, default=5000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_norm', type=float, default = 10, metavar='N',
                        help='the total norm')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    word_num = len(dictionary.keys())
    model = RewardModel(word_num, rnn_size=args.rnn_size,feature_size= args.feature_size,
                        num_layers = args.num_layers,bidirectional= args.brnn, dropout=False)
    device = 0
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda(device)
        print('using cuda device {}'.format(device))
    else:
        print('using cpu.')

    weightspath = os.path.join(modelfolder, 'weights.pth')
    if args.reuse_weigths and os.path.exists(weightspath):
        weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights_dict)
        print('load check points from {}'.format(weightspath))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                          nesterov =  True, weight_decay=args.weight_decay)
    #optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=args.momentum)
    steps, vals = [], []
    with h5py.File(h5dataFile,'r') as h5_data:
        data_loader = Dataloader(h5_data, batch_size=  args.batch_size, dictionary= dictionary,
                                 split_dict = split_dict, refer_dict = refer_dict)
        train_flow = data_loader.get_flow('train')
        
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)
        
        batch_count = 0
        model.train()
        for epochNumber in range(args.maxepoch):

            for (padded_left, padded_right ), pair_label, \
                (lengths_left, lengths_right), (_,_) in train_flow:
                inputs = ((padded_left, padded_right ), (lengths_left, lengths_right))
                pair_label = to_device(pair_label, model.device_id)
                pred = model(inputs)
                loss = nn.SmoothL1Loss()(pred, pair_label)
                #loss = 0.5*torch.mean((pred-pair_label)**2)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm(model.parameters(), args.max_norm, norm_type=2)
                
                optimizer.step()
                batch_count += 1

                loss_val = loss.data.cpu().numpy().mean()
                steps.append(batch_count)
                vals.append(float(loss_val))
                if batch_count % args.showfre == 0:
                    step = min(1, len(steps))
                    display_loss(steps[0::step], vals[0::step], plot=None, name= 'reward model')
                    steps[:] = []
                    vals[:]  = []
                    print('epoch {}, batch_acount {}, loss {}, grad norm {}\n'.
                           format(epochNumber, batch_count, loss_val,grad_norm))
                if np.mod(batch_count, args.savefreq) == 0:
                    torch.save(model.state_dict(), weightspath)
            cur_weights = model.state_dict()
            torch.save(cur_weights, weightspath)
                    

