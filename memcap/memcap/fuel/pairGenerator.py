import numpy as np
from copy import copy
from torch.autograd import Variable
import random
from ..proj_utils.local_utils import pre_process_img, split_words

class SimplePairData():
    def __init__(self, data, Index, dictionary=None, batch_size=128, mode='channel',
                 norm=True, cuda=True):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        '''
        self.__dict__.update(locals())
        self.images = self.data['images']
        self.sentence = self.data['sentences'].value

        filenames = self.data['filenames'].value
        self.ident = []
        for name in filenames:
            name = name.decode() if type(name) is bytes else name
            self.ident.append(self.get_identifier(name))

        self.Totalnum = len(Index)
        self.n_words = len(dictionary.keys())
        self.reset()

    def get_identifier(self, name):
        idx = name.find('augment')
        if idx == -1:
            return name
        else:
            ident = name[0:idx - 1]
            return ident


    def reset(self):
        self.chunkstart = 0

        sample_rand_Ind = copy(self.Index)
        random.shuffle(sample_rand_Ind)
        self.totalIndx = sample_rand_Ind

        self.numberofchunk = (self.Totalnum + self.batch_size - 1) // self.batch_size  # the floor
        self.chunkidx = 0

    def sent2ind(self, sent):
        return [self.dictionary[w] if self.dictionary[w] < self.n_words else 1 for w in split_words(sent)]

    def pad_mask(self, seqs, lengths):
        '''
        :param seqs: the list of sequences
        :param lengths:   the len of each sequence,including the <eos>
        :return: padded seq and mask.
        '''
        n_samples = len(lengths)
        maxlen = np.max(lengths)
        # because we need to make it has one end of sentence in the end, so one more symbol.
        caps_x = np.zeros((maxlen, n_samples)).astype('int64')
        caps_mask = np.zeros((maxlen, n_samples)).astype('float32')
        for ind, s in enumerate(seqs):
            caps_x[:lengths[ind]-1, ind] = s
            caps_mask[:lengths[ind], ind] = 1.

        return caps_x, caps_mask


    def next(self):
        return self.__next__()

    def __next__(self):

        thisnum = min(self.batch_size, self.Totalnum - self.chunkidx * self.batch_size)
        curr_indices = self.totalIndx[self.chunkstart: self.chunkstart + thisnum]

        self.chunkstart += thisnum
        self.chunkidx += 1
        seqs_left = []
        seqs_right = []
        pair_label = []

        if self.chunkidx > self.numberofchunk:
            self.reset()
            raise StopIteration()

        for ind, idx in enumerate(curr_indices):
            this_sentlist = self.sentence[idx]
            total_caps = len(this_sentlist)
            cand_ind_list = list(range(0, total_caps))
            this_ind = random.randint(0, total_caps-1)
            cand_ind_list.pop(this_ind)
            left_sent = this_sentlist[this_ind]
            if np.random.random() < 0.5: # we take one sample from this batch
                this_ind = random.randint(0, total_caps-2)
                right_sent = this_sentlist[cand_ind_list[this_ind]]
                this_label = 1
            else:
                assert self.batch_size!= 1, 'you cannot use batch size 1 in this case'
                this_label = -1

                while True:
                    right_ind = random.randint(0, self.Totalnum-1)
                    right_index = self.totalIndx[right_ind]
                    #print(len(self.ident), len(self.totalIndx), idx ,right_index)
                    if self.ident[idx] != self.ident[right_index]:
                        break
                right_sentlist = self.sentence[right_index]
                right_randInd = random.randint(0, len(right_sentlist)-1)
                right_sent = right_sentlist[right_randInd]

            left_sent = left_sent.decode() if type(left_sent) is bytes else left_sent
            right_sent = right_sent.decode() if type(right_sent) is bytes else right_sent
            seqs_left.append(self.sent2ind(left_sent))
            seqs_right.append(self.sent2ind(right_sent))
            pair_label.append(this_label)

        lengths_left  = [len(s)+1 for s in seqs_left]
        lengths_right = [len(s)+1 for s in seqs_right]

        padded_left, mask_left   =  self.pad_mask(seqs_left,  lengths_left)
        padded_right, mask_right =  self.pad_mask(seqs_right, lengths_right)

        
        return (padded_left, padded_right ), pair_label, (lengths_left, lengths_right), (mask_left,mask_right)
        
    def __iter__(self):
        return self

class PairData():
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
            self.train_cls = SimplePairData(self.data, self.train, self.dictionary,
                                            self.batch_size, cuda=self.cuda)
            return self.train_cls

        if split == 'test':
            self.test_cls = SimplePairData(self.data, self.test, self.dictionary,
                                           self.batch_size, cuda=self.cuda)
            return self.test_cls

        if split == 'valid':
            self.valid_cls = SimplePairData(self.data, self.valid, self.dictionary,
                                            self.batch_size, cuda=self.cuda)
            return self.valid_cls

        if split == 'all':
            self.all_cls = SimplePairData(self.data, self.all, self.dictionary,
                                          self.batch_size, cuda=self.cuda)
            return self.all_cls