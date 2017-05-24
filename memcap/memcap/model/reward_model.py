import torch.nn as nn
import math
import torch
from torch.nn import functional as F

from collections import namedtuple
from torch.autograd import Variable
from ..proj_utils.torch_utils import *
import torch.optim as optim

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

def sort_batch(data, seq_len):
    '''
    data: (B*T*dim)
    '''
    batch_size = data.size(0)
    sorted_seq_len, sorted_idx  = torch.sort(seq_len, dim=0, descending=True)
    sorted_data = data[sorted_idx]
    _, reverse_idx  = torch.sort(sorted_idx, dim=0, descending=False)
    return sorted_data, sorted_seq_len, reverse_idx

def two_ind(data, first, second):
    '''
    data: 3 dimension
    first: 1 d tensor
    second: 1 d tensor
    '''
    first  = to_device(first, data, False)
    second = to_device(second, data, False)
    row, col, dim = data.size()
    index = first*col + second
    last_out = data.contiguous().view(-1, dim)[index]
    return last_out

def get_last(data, lens_th):
    '''
    data: time*batch*dim
    len_list: length of batch, actually time for each batch
    return: batch*dim
    '''
    first   = lens_th -1
    second  = torch.arange(0, len(lens_th)).long()
    #print(data.size(), first, second)
    return two_ind(data, first, second)

def print_tensor(tensor):
    print(torch.max(torch.abs(tensor)))
    print(tensor)

class GRU(nn.Module):
    def __int__(self, word_num, rnn_size=128,bidirectional= False, dropout=True):
        self.__dict__.update(locals())
        super(GRU, self).__init__()
        self.WX = nn.Parameter()



class RewardModel(nn.Module):

    def __init__(self,word_num, rnn_size=128,feature_size=64 ,num_layers = 2,
                 emb_dim=64, bidirectional= False, dropout=True):
        self.__dict__.update(locals())
        super(RewardModel, self).__init__()
        input_size = word_num
        self.register_buffer('device_id', torch.zeros(1))
        self.one_hot = one_hot(self.word_num)
        self.emb_model = nn.Embedding(self.word_num, self.rnn_size)

        self.encoder = nn.GRU(rnn_size, rnn_size,
                            num_layers=num_layers,
                            dropout=dropout, batch_first= True,
                            bidirectional=bidirectional)
        #self.encoder.weight_hh_l0.register_hook(print_tensor)
        self.encoder_dim = 2*rnn_size if bidirectional else rnn_size

        self.encoder2hid = nn.Linear(self.encoder_dim, self.feature_size)

        #please note that even if the input can be batch_first,
        # the output can only be time*batch*dim                     

    def forward(self, inputs, hidden=None):
        (ldata, rdata), (llen, rlen) = inputs
        llen, rlen = to_device(llen, self.device_id, False).long(), to_device(rlen, self.device_id, False).long()

        # ldata, rdata = torch.LongTensor(ldata), torch.LongTensor(rdata)
        #ldata, rdata = self.one_hot(ldata), self.one_hot(rdata)
        ldata, rdata = to_device(ldata, self.device_id), to_device(rdata, self.device_id)
        ldata, rdata = self.emb_model(ldata), self.emb_model(rdata)

        # the following make the code Batch dimension first
        ldata = ldata.transpose(0, 1)
        rdata = rdata.transpose(0, 1)

        left_data, left_len, left_rev = sort_batch(ldata, llen)
        right_data, right_len, right_rev = sort_batch(rdata, rlen)
        left_data = to_device(left_data, self.device_id)
        right_data = to_device(right_data, self.device_id)
        left_rev = to_device(left_rev, self.device_id, var=False)
        right_rev = to_device(right_rev, self.device_id, var=False)

        left_pack = pack(left_data, list(left_len), batch_first=True)
        right_pack = pack(right_data, list(right_len), batch_first=True)

        left_outputs, left_hidden_t = self.encoder(left_pack, hidden)
        right_outputs, right_hidden_t = self.encoder(right_pack, hidden)

        left_unpack_out = unpack(left_outputs)[0]
        right_unpack_out = unpack(right_outputs)[0]

        left_last_out  = get_last(left_unpack_out, left_len)
        right_last_out = get_last(right_unpack_out, right_len)

        org_left_last = left_last_out[left_rev]
        org_right_last = right_last_out[right_rev]

        left_encoder_hid = F.tanh(self.encoder2hid(org_left_last))
        right_encoder_hid = F.tanh(self.encoder2hid(org_right_last))

        cos_sim = cosine_similarity(left_encoder_hid, right_encoder_hid)

        return cos_sim


    def forward_(self, inputs,  hidden=None):
        (ldata, rdata), (llen, rlen) = inputs
        llen, rlen = to_device(llen, self.device_id).long(), to_device(rlen, self.device_id).long()

        #ldata, rdata = torch.LongTensor(ldata), torch.LongTensor(rdata)
        #ldata, rdata = self.one_hot(ldata), self.one_hot(rdata)
        ldata, rdata = to_device(ldata, self.device_id), to_device(rdata, self.device_id)
        ldata, rdata = self.emb_model(ldata), self.emb_model(rdata)
        # the following make the code Batch dimension first
        #left_data = ldata.transpose(0,1)
        #right_data = rdata.transpose(0,1)

        left_data  = to_device(ldata, self.device_id)
        right_data = to_device(rdata, self.device_id)

        left_outputs,  left_hidden_t  = self.encoder(left_data,  hidden)
        right_outputs, right_hidden_t = self.encoder(right_data, hidden)

        left_last_out  = get_last(left_outputs,  llen)
        right_last_out = get_last(right_outputs, rlen)
        #print(self.encoder.weight_hh_l0)
        left_encoder_hid  = F.tanh(self.encoder2hid(left_last_out))
        right_encoder_hid = F.tanh(self.encoder2hid(right_last_out))

        cos_sim = cosine_similarity(left_encoder_hid, right_encoder_hid)

        return cos_sim