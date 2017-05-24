import cPickle as pkl
import os
from collections import OrderedDict
from preprocessing.sequence import pad_sequences
import numpy as np
import re
from Core.utils_func import split_words, expand_list
from utils.local_utils import *
import json
from backend.export import npwrapper
from fuel.Extractor import get_cnn_img
import random

def prepare_data(caps, features, worddict, maxlen=None, n_words=1000, zero_pad=False, online_feature=False):
    """
    Parameters:
    -----------
    caps:           list of annocations (['this is a good example.','asas'],0)] 
                    the second term denotes the feature index.abs
    features:       Tensor (nsample, row, col, channel)
    worddict:       dictionary of words to number {'word': 1}
    maxlen:         max length of captions.
    n_words:        max number of dictioanry. if beyond set to unknown.
    zero_pad:       pad the feature map to given shape.
    online_feature: if True, return image, else return features.

    Returns:
    --------
    caps_x:     (sequence_len, n_samples)
    caps_mask:  (sequence_len, n_samples)
    feat:       (n_sample, chennel, flat_position)
    """
    seqs = []
    feat_list = []
    for cc in caps:
        total_caps = len(cc[0])
        sentence = cc[0][random.randint(0, total_caps-1)]
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in split_words(sentence)])
        feat_list.append(features[cc[1]: cc[1] + 1])

    lengths = [len(s) for s in seqs]
    if maxlen != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None, None
    
    if online_feature == False:
        feat = np.zeros((len(feat_list), feat_list[0].shape[1])).astype('float32')
        for idx, ff in enumerate(feat_list):
            feat[idx,:] = np.array(ff.todense())
        feat = feat.reshape([feat.shape[0], 512,14*14])
        feat = np.transpose(feat, (0, 2, 1))

        if zero_pad:
            feat_pad = np.zeros((feat.shape[0], feat.shape[1]+1, feat.shape[2])).astype('float32')
            feat_pad[:,:-1,:] = feat
            feat = feat_pad
    else:
        dest_shape = (224,224,3)
        shape = (3,224,224)
        feat = np.zeros((len(feat_list),) + shape).astype('float32')
       
        for idx, ff in enumerate(feat_list):
            #y[idx,:] = np.array(ff)
            ff = ff[0]
            feat[idx,:] = get_cnn_img(ff,dest_shape, norm = True)
        #print feat.shape
    n_samples = len(seqs)
    maxlen = np.max(lengths)+1
    #because we need to make it has one end of sentence in the end, so one more symbol.
    caps_x = np.zeros((maxlen, n_samples)).astype('int64')
    caps_mask = np.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        caps_x[:lengths[idx],idx] = s
        caps_mask[:lengths[idx]+1,idx] = 1.

    #caps, caps_mask = pad_sequences(seqs, maxlen=None, dtype='int64',
    #             padding='post', truncating='pre', value=0.)
    #caps = np.transpose(caps, (1,0))
    #caps_mask = np.transpose(caps_mask, (1,0))

    return npwrapper(caps_x), npwrapper(caps_mask), npwrapper(feat)

from Extractor import ListExtractor as dataExtractor
import cPickle

def get_anno(classparams, filepath = None, chunknum = 5000, batchsize=32, get_img=True):
    StruExtractor = dataExtractor(classparams)
    datainfo = StruExtractor.datainfo
    Totalnum = datainfo['Totalnum']
    
    totalIndx = np.arange(Totalnum)
    numberofchunk = (Totalnum + chunknum - 1)// chunknum   # the floor
    chunkstart = 0 
    
    thisanno = [None for _ in range(chunknum)]
    Total_anno = []
    for chunkidx in range(numberofchunk):
        thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum

        _, returned_anno = StruExtractor.getImg_Anno(thisInd, thisanno, get_img=False)
        print returned_anno[0:10]
        Total_anno.extend(returned_anno[0:thisnum]) #thisanno is changed, but extend already copied it's content as long ass it has no mutable ele'
    if filepath is not None:
        with open(filepath, 'wb') as f:
            cPickle.dump(Total_anno, f,protocol=cPickle.HIGHEST_PROTOCOL)
    return Total_anno
    
def group_cap(cap):
    returned_cap = []
    orderedCap = OrderedDict()
    for thiscap in cap:
        ind = thiscap[1]
        thiswords = thiscap[0]
        if ind not in orderedCap:
            orderedCap[ind] = thiswords
        else:
            orderedCap[ind] = orderedCap[ind] + ' ' + thiswords
    for key in  orderedCap.keys():
        returned_cap.append( (orderedCap[key], key)  )
    return returned_cap

def load_data(load_train=True, load_dev= True, load_test= True, root_path='../Data/TrainingData/bladder', 
              img_ext='.png', online_feature=False):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset
    '''
    print '... loading data. ...'
    from scipy import sparse
    
    cap_root_path = os.path.join(root_path, 'Feat_conv')
    img_root_path = os.path.join(root_path, 'Img')
    anno_root_path  = os.path.join(root_path, 'Annotation')
    feat_len = 512*14*14
    train_cap = []

    train_cap_pkl = os.path.join(cap_root_path , 'bladder_align_train_cap.pkl')
    test_cap_pkl = os.path.join(cap_root_path  , 'bladder_align_test_cap.pkl')
    valid_cap_pkl = os.path.join(cap_root_path , 'bladder_align_valid_cap.pkl')

    trainingSplitFile = os.path.join(root_path , 'train_list.json')
    testingSplitFile = os.path.join(root_path  , 'test_list.json')
    validationSplitFile = os.path.join(root_path , 'valid_list.json')


    classparams = {}
    classparams['datadir']   =  img_root_path 
    classparams['annodir']  =   anno_root_path
    classparams['dataExt']   =  ['.png']             # the data ext
    classparams['destin_shape']   =  (224,224) 
    classparams['channel']   =  3

    split_file_list = [trainingSplitFile,testingSplitFile ,validationSplitFile ]
    file_path_list = [train_cap_pkl, test_cap_pkl, valid_cap_pkl]

    for split_file, filepath in zip(split_file_list, file_path_list):
        this_classparams = classparams.copy()
        if not os.path.exists(filepath):
            with open(split_file) as data_file:    
                SplitDict = json.load(data_file)
                file_list =  SplitDict['img']
            this_classparams['nameList']  =  file_list
            get_anno(this_classparams, filepath=filepath, chunknum = 5000, batchsize=32, get_img=False)

    if load_train:    
        with open(train_cap_pkl, 'rb') as ft_cap: 
            train_cap = pkl.load(ft_cap)
            returned_train_cap = train_cap #group_cap(train_cap)

        if online_feature == False:
            feat_root_path = cap_root_path
            train_feat = None
            #train_feat = np.ones((len(returned_train_cap), feat_len))
            train_feat = sparse.csr_matrix((0, feat_len))
            current = 0
            for idx in range(9):
                print(idx)
                filename = 'bladder_align_train' + str(idx) + '.pkl'           
                with open(os.path.join(feat_root_path, filename), 'rb') as ft_feat:
                    #train_feature_list.append(pkl.load(ft_feat).todense())
                    thistemp = pkl.load(ft_feat)
                    train_feat =  sparse.vstack((train_feat, thistemp))
                    current = current + thistemp.shape[0]       
            assert current == len(returned_train_cap)
        else:
            train_feat = []
            for cap_tuple in returned_train_cap:
                name = cap_tuple[2]
                train_feat.append(os.path.join(img_root_path, name+img_ext))
        
        train = (returned_train_cap, train_feat)
    else:
        train = None
    
    #------Load testing set-----------------

    if load_test:
        test_cap = []
        with open(test_cap_pkl, 'rb') as ft_cap:
            test_cap = pkl.load(ft_cap)
        returned_test_cap = test_cap #group_cap(test_cap)

        if online_feature == False:
            feat_root_path = cap_root_path
            with open(os.path.join(feat_root_path,'bladder_align_test0.pkl'), 'rb') as f:
                test_feat = pkl.load(f)
            returned_test_cap = group_cap(test_cap)  
        else:     
            test_feat = []
            for cap_tuple in returned_test_cap:
                name = cap_tuple[2]
                test_feat.append(os.path.join(img_root_path, name+img_ext))

        test = (returned_test_cap, test_feat)

    else:
        test = None

    valid_cap = []    
    if load_dev:
        with open(valid_cap_pkl, 'rb') as ft_cap:
            valid_cap = pkl.load(ft_cap)
        returned_valid_cap = valid_cap #group_cap(valid_cap)
        
        if online_feature == False:
            feat_root_path = cap_root_path
            valid_feat = sparse.csr_matrix((0, feat_len))
            for idx in range(1):
                filename = 'bladder_align_test' + str(idx) + '.pkl'
                with open(os.path.join(feat_root_path, filename), 'rb') as ft_feat:
                    #valid_feature_list.append(pkl.load(ft_feat).todense())
                    thistemp = pkl.load(ft_feat)
                    valid_feat =  sparse.vstack((valid_feat, thistemp))
        else:
            valid_feat = []
            for cap_tuple in returned_valid_cap:
                name = cap_tuple[2]
                valid_feat.append(os.path.join(img_root_path, name+img_ext))

        valid = (returned_valid_cap, valid_feat)
    else:
        valid = None
    
    #with open(os.path.join(cap_root_path,'dictionary.pkl'), 'rb') as f:
    #    worddict = pkl.load(f)

    dict_path = os.path.join(cap_root_path,'dictionary.pkl')
    worddict = OrderedDict()
    worddict['<eos>'] = 0
    worddict['UNK'] = 1
    wordIndex = 2

    total_cap= expand_list(returned_train_cap + returned_valid_cap + returned_test_cap)
    for this_cap_tuple in total_cap:
        this_cap_list = this_cap_tuple[0]
        for this_cap in this_cap_list:
            words = split_words(this_cap)
            for k in words:
                if k not in worddict:
                    worddict[k] = wordIndex
                    wordIndex = wordIndex + 1
    print worddict
    with open(dict_path, 'wb') as f:
        pkl.dump(worddict, f, protocol=pkl.HIGHEST_PROTOCOL)

    return train, valid, test, worddict
