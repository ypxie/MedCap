from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import skimage
import skimage.io
import scipy.misc
from multiprocessing.dummy import Process, Queue, Pool

class DataLoader():

    def reset_iterator(self, split):
        self._prefetch_process[split].terminate()
        self._prefetch_process[split].join()
        self._prefetch_process[split] = BlobFetcher(split, self)
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_h5, opt.input_att_h5, opt.input_label_h5)
        self.h5_fc_file = h5py.File(self.opt.input_fc_h5, 'r')
        self.h5_att_file = h5py.File(self.opt.input_att_h5, 'r')
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')


        # extract image size from dataset
        fc_size = self.h5_fc_file['fc'].shape
        att_size = self.h5_att_file['att'].shape
        assert fc_size[0] == att_size[0], 'fc and att same numer'
        self.num_images = fc_size[0]
        print('read %d image features' %(self.num_images))

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self)
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                self._prefetch_process[split].terminate()
                self._prefetch_process[split].join()
        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = np.ndarray((batch_size * seq_per_img,) + self.h5_fc_file['fc'].shape[1:], dtype = 'float32')
        att_batch = np.ndarray((batch_size * seq_per_img,) + self.h5_att_file['att'].shape[1:], dtype = 'float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            import time
            t_start = time.time()
            # fetch image
            #tmp_fc, tmp_att, tmp_label, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch[i * seq_per_img:(i+1) * seq_per_img], \
                att_batch[i * seq_per_img:(i+1) * seq_per_img], \
                ix, tmp_wrapped = self._prefetch_process[split].get()

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
                for q in range(seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.dataloader.h5_label_file['labels'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
            
            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix] - 1])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
            #print(i, time.time() - t_start)

        # generate mask
        t_start = time.time()
        nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        #print('mask', time.time() - t_start)

        data = {}
        data['fc_feats'] = fc_batch
        data['att_feats'] = att_batch
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch 
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader

        self.pool = Pool(4)
        self.fifo = []

    # Add more in the queue
    def reset(self):
        if len(self.fifo) == 0:
            self.cur_idx = self.dataloader.iterators[self.split]
        split_ix = self.dataloader.split_ix[self.split]
        for i in xrange(512 - len(self.fifo)):
            ix = split_ix[self.cur_idx]
            if self.cur_idx + 1 >= len(split_ix):
                self.cur_idx = 0
            else:
                self.cur_idx += 1
            self.fifo.append(self.pool.apply_async(self._get_minibatch, (ix, )))

    def terminate(self):
        self.pool.terminate()

    def join(self):
        self.pool.join()

    def _get_next_minibatch_inds(self):
        split_ix = self.dataloader.split_ix[self.split]
        max_index = len(split_ix)
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next
        ix = split_ix[ri]

        return ix, wrapped

    def _get_minibatch(self, ix):
        wrapped = False
        if ix == self.dataloader.split_ix[self.split][-1]:
            wrapped = True

        return (self.dataloader.h5_fc_file['fc'][ix, :].astype('float32'),
            self.dataloader.h5_att_file['att'][ix, :, :, :].astype('float32'),
            ix,
            wrapped)

    def get(self):
        if len(self.fifo) < 400:
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.fifo.pop(0).get()

        assert tmp[2] == ix, "ix not equal"
        assert tmp[3] == wrapped, "wrapped not equal"

        return tmp