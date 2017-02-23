import sys
import os
import argparse

parser = argparse.ArgumentParser(description = 'Bladder Classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N'
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N'
                    help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
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

from proj_utils.Extractor import ListExtractor as dataExtractor
from train_eng import get_mean_std
from proj_utils.local_utils import *
from utils.data_augmentor import ImageGenerator

import json
trainingSplitFile = os.path.join(dataroot, 'train_list.json')
testingSplitFile = os.path.join(dataroot,  'valid_list.json')

with open(trainingSplitFile) as data_file:    
    trainingSplitDict = json.load(data_file)
with open(testingSplitFile) as data_file:    
    testingSplitDict = json.load(data_file)
train_file_list =  trainingSplitDict['img']
train_label_list = trainingSplitDict['label']
test_file_list =  testingSplitDict['img']
test_label_list = testingSplitDict['label']

if  __name__ == '__main__':
    nb_class = 4

    loss = 'categorical_crossentropy'
    use_weighted = 0    #means you have a weight mask in the last dimension of mask

    # for build the model    
    img_channels = 3    
    lr = 3e-3
    weight_decay = 1e-6

    batch_size = 2
    meanstd = 0   # it is important to set this as 0 for FCN
    chunknum = int(10000)
    maxepoch = 1280
    matrefresh = 10
    maxInnerEpoch = 10
    meanstdrefresh = 1
    refershfreq = 10
    savefre = 1  # np.mod(chunidx, savefre) == 0:
    
    classparams = {}
    classparams['datadir']   =  trainingimagefolder
    classparams['dataExt']   =  ['.png']             # the data ext
    classparams['nameList']  =  train_file_list
    classparams['labelList'] =  train_label_list 
    classparams['destin_shape']   =  (224,224) 
    classparams['channel']   =  img_channels
    StruExtractor = dataExtractor(classparams)
    
    testing_classparams = classparams.copy()
    testing_classparams['nameList']  =  test_file_list
    testing_classparams['labelList'] =  test_label_list 
    TestingExtractor = dataExtractor(testing_classparams)
    testing_data, testing_label = TestingExtractor.getOneDataBatch_stru()


    rebuildmodel = 1
    reuseweigths = 1
    show_progress = 0 #if you want to show the testing cases.
    if not os.path.exists(modelfolder):
        os.makedirs(modelfolder)
    modelDict = {}
    modelpath = os.path.join(modelfolder, 'strumodel.h5') #should include model and other parameters
    weightspath = os.path.join(modelfolder,'weights.h5')
    best_weightspath = os.path.join(modelfolder,'best_weights.h5')
    arctecurepath = os.path.join(modelfolder,'arc.json')
    matpath = os.path.join(modelfolder,'matinfo.h5')
    meanstdpath = os.path.join(modelfolder, 'meanstd.h5')
    paramspath = os.path.join(modelfolder, 'params.h5')
    
    if not os.path.isfile(arctecurepath) or  rebuildmodel == 1:
       strumodel = buildmodel(img_channels,lr = lr, activ = activ,loss = loss,nb_class = 4)
       if reuseweigths == 1 and os.path.isfile(weightspath):
          strumodel.load_weights(best_weightspath )# 12)

    Matinfo = StruExtractor.getMatinfo() # call this function to generate nece info
       
    datainfo = Matinfo['datainfo']

    meanstdDic = {}
    if not os.path.isfile(meanstdpath) or meanstdrefresh == 1:
       thismean, thisdev = get_mean_std(StruExtractor, meanstd)
       meanstdDic['thismean'] = thismean
       meanstdDic['thisdev'] = thisdev
       dd.io.save(meanstdpath, meanstdDic, compression='zlib')
    else:
       meanstdDic = dd.io.load(meanstdpath)
       thismean = meanstdDic['thismean']
       thisdev = meanstdDic['thisdev']


    thisbatch = np.zeros((chunknum,) + tuple(datainfo['inputshape'] ))
    thislabel = np.zeros((chunknum,) + tuple(datainfo['outputshape']))

    
    print('finish compiling!')
    best_score = 0
    for epochNumber in range(maxepoch):

      if np.mod(epochNumber+1, refershfreq) == 0:
        Matinfo = StruExtractor.getMatinfo() # call this function to generate nece info
        thismean, thisdev = get_mean_std(StruExtractor, meanstd)
        datainfo = Matinfo['datainfo']
      Totalnum = datainfo['Totalnum']
      totalIndx = np.random.permutation(np.arange(Totalnum))

      numberofchunk = (Totalnum + chunknum - 1)// chunknum   # the floor
      chunkstart = 0

      progbar = generic_utils.Progbar(Totalnum*maxInnerEpoch)
      for chunkidx in range(numberofchunk):
          thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
          thisInd = totalIndx[chunkstart: chunkstart + thisnum]
          StruExtractor.getOneDataBatch_stru(thisInd, thisbatch[0:thisnum], thislabel[0:thisnum])
          chunkstart += thisnum
          BatchData = thisbatch[0:thisnum].astype(K.FLOATX)
          BatchLabel = thislabel[0:thisnum].astype(K.FLOATX)

          #---------------Train your model here using BatchData------------------
          BatchData -= thismean
          BatchData /= thisdev
   
          print('Training--Epoch--%d----chunkId--%d', (epochNumber, chunkidx))
          for innerEpoch in range(maxInnerEpoch):
                print('innerEpoch {s},processing {d} samples'.format(s=str(innerEpoch), d=str((innerEpoch+1)*chunknum)))
                #for X_batch, Y_batch in datagen.flow(BatchData, BatchLabel, batch_size ):
                for X_batch, Y_batch in batchflow(batch_size, BatchData ,BatchLabel ):
                    loss = strumodel.train_on_batch({'input': X_batch}, {'output': Y_batch})
                    if type(loss) == list:
                        loss = loss[0]
                    assert not np.isnan(loss) ,"nan error"
                
                loss, acc = strumodel.evaluate(testing_data, testing_label, verbose=0)
                print('\nTesting loss: {}, acc: {}, best_score: {}'.format(loss, acc, best_score))
                if acc >=  best_score:
                    best_score = acc
                    print('update to new best_score: {}'.format(best_score))
                    best_weight = strumodel.get_weights()
                    strumodel.save_weights(best_weightspath,overwrite = 1)
                elif best_score - acc > 0.03: 
                    strumodel.set_weights(best_weight)
		        print('weights have been reset to best_weights!')
                       
                progbar.add(BatchData.shape[0], values = [("train loss", loss)])
                json_string = strumodel.to_json()
                open(arctecurepath, 'w').write(json_string)
                strumodel.save_weights(weightspath,overwrite = 1)

      strumodel.save_weights(weightspath,overwrite = 1)
      dd.io.save(paramspath, classparams, compression='zlib')
