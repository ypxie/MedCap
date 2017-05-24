import numpy as np
from copy import copy
from torch.autograd import Variable
import random
from ..memcap.proj_utils.local_utils import pre_process_img

class SimpleCapData():
    def __init__(self, data, Index, batch_size=128, mode='channel',
                 norm=True, cuda=True):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        '''
        self.__dict__.update(locals())

        self.images = self.data['images']
        self.sentence = self.data['sentences'].value

        self.Totalnum = len(Index)

        self.batch_data = np.zeros((batch_size,) + self.images.shape[1::], dtype=np.float32)
        #self.batch_label = np.zeros((batch_size,) + self.conclusion.shape[1::], dtype=np.float32)

        self.reset()

    def reset(self):
        self.chunkstart = 0

        sample_rand_Ind = copy(self.Index)
        random.shuffle(sample_rand_Ind)
        self.totalIndx = sample_rand_Ind

        self.numberofchunk = (self.Totalnum + self.batch_size - 1) // self.batch_size  # the floor
        self.chunkidx = 0

    def __next__(self):

        thisnum = min(self.batch_size, self.Totalnum - self.chunkidx * self.batch_size)
        curr_indices = self.totalIndx[self.chunkstart: self.chunkstart + thisnum]

        self.chunkstart += thisnum
        self.chunkidx += 1
        #idx_list = []
        seqs = []
        for ind, idx in enumerate(curr_indices):
            self.batch_data[ind] = pre_process_img(self.images[idx], yuv=False,
                                                   norm=self.norm, mode=self.mode)
            this_sentlist = self.sentence[idx]
            total_caps = len(this_sentlist)
            this_sent = this_sentlist[random.randint(0, total_caps-1)]
            seqs.append([self.dictionary[w] if self.dictionary[w] < self.n_words else 1 for w in split_words(this_sent)])
        lengths = [len(s)+1 for s in seqs]
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        # because we need to make it has one end of sentence in the end, so one more symbol.
        caps_x = np.zeros((maxlen, n_samples)).astype('int64')
        caps_mask = np.zeros((maxlen, n_samples)).astype('float32')
        for ind, s in enumerate(seqs):
            caps_x[:lengths[ind]-1, ind] = s
            caps_mask[:lengths[ind], ind] = 1.

        if self.chunkidx > self.numberofchunk:
            self.reset()
            raise StopIteration()

        return self.batch_data[0:thisnum].astype(np.float32),caps_x, caps_mask, lengths

    def __iter__(self):
        return self

class CapData():
    def __init__(self, data, batch_size=128, split_dict=None,
                 refer_dict=None, dictionary=None, cuda=False):
        '''
        Construct train, valid and test iterator

        Parameters:
        -----------
        data: a hdf5 opened pointer
        split_dict:  dictionary{train:[list of file name], test:[], valid: []} 
        refer_dict:  list of dictionary of {filename:ind}, used to get_split_ind
        dictionary: {word:id}
        '''

        self.__dict__.update(locals())

        self.get_split_ind()

    def get_split_ind(self):
        '''
        transfer split file name list to list of index.
        '''
        self.train = []
        self.test = []
        self.valid = []
        self.all = []

        for name in self.split_dict.get('train', []):
            self.train.append(self.refer_dict[name])

        for name in self.split_dict.get('test', []):
            self.test.append(self.refer_dict[name])

        for name in self.split_dict.get('valid', []):
            self.valid.append(self.refer_dict[name])

        for name in self.split_dict.get('all', []):
            self.all.append(self.refer_dict[name])

    def get_flow(self, split='train'):
        if split == 'train':
            self.train_cls = SimpleCapData(self.data, self.train, self.batch_size, cuda=self.cuda)
            return self.train_cls

        if split == 'test':
            self.test_cls = SimpleCapData(self.data, self.test, self.batch_size, cuda=self.cuda)
            return self.test_cls

        if split == 'valid':
            self.valid_cls = SimpleCapData(self.data, self.valid, self.batch_size, cuda=self.cuda)
            return self.valid_cls

        if split == 'all':
            self.all_cls = SimpleCapData(self.data, self.all, self.batch_size, cuda=self.cuda)
            return self.all_cls