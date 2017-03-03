import numpy as np
from copy import copy
from torch.autograd import Variable
import random
from proj_utils.local_utils import pre_process_img

class SimpleData():
    def __init__(self, data, Index, batch_size=128, mode = 'channel', 
                 norm = True):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        '''
        
        self.__dict__.update(locals())
        
        self.images = self.data['images']
        self.conclusion = self.data['conclusion']

        self.Totalnum = len(Index)

        self.batch_data = np.zeros( (batch_size,) + self.images.shape[1::])
        self.batch_label = np.zeros((batch_size,) + self.conclusion.shape[1::])

        self.reset()

    def reset(self):
        self.chunkstart = 0
        
        sample_rand_Ind = copy(self.Index)
        random.shuffle(sample_rand_Ind)
        self.totalIndx = sample_rand_Ind

        self.numberofchunk = (self.Totalnum + self.batch_size - 1) // self.batch_size   # the floor
        self.chunkidx = 0
    
    def next(self):
        
        thisnum = min(self.batch_size, self.Totalnum - self.chunkidx*self.batch_size)
        curr_indices = self.totalIndx[self.chunkstart: self.chunkstart + thisnum]
        
        self.chunkstart += thisnum
        self.chunkidx += 1

        for ind, idx in enumerate(curr_indices):
            self.batch_data[ind] = pre_process_img(self.images[idx],yuv = False,
                                                   norm=self.norm, mode = self.mode)
            self.batch_label[ind] = self.conclusion[idx]
        
        if self.chunkidx > self.numberofchunk:
            self.reset()
            raise StopIteration()

        return self.batch_data[0:thisnum] , self.batch_label[0:thisnum]

    def __iter__(self):
        return self



class CNNData():
    def __init__(self, data, batch_size=128, split_dict= None, 
                 refer_dict = None):
        '''
        Construct train, valid and test iterator

        Parameters:
        -----------
        data: a hdf5 opened pointer
        split_dict:  dictionary{train:[list of file name], test:[], valid: []} 
        refer_dict:  dictionary of {filename:ind}, used to get_split_ind

        '''
        self.__dict__.update(locals())

        self.get_split_ind()
        
        self.train_cls = SimpleData(self.data, self.train, self.batch_size)
        self.test_cls  = SimpleData(self.data, self.test, self.batch_size)
        self.valid_cls = SimpleData(self.data, self.valid, self.batch_size)


    def get_split_ind(self):
        '''
        transfer split file name list to list of index.
        '''
        self.train = []
        self.test  = []
        self.valid = []

        for name in self.split_dict.get('train', []):
            self.train.append(self.refer_dict[name])
        
        for name in self.split_dict.get('test', []):
            self.test.append(self.refer_dict[name])
        
        for name in  self.split_dict.get('valid', []):
            self.valid.append(self.refer_dict[name])
        
        
    def get_flow(self, split='train'):
        if split == 'train':
           return self.train_cls

        if split == 'test':
           return self.test_cls
           
        if split == 'valid':
           return self.valid_cls   