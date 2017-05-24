import numpy as np
from copy import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
try:
    from visdom import Visdom
except:
    print('Better install visdom')

def to_variable(x, requires_grad=False, cuda=False):
    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    if cuda:
        return Variable(x, requires_grad=requires_grad).cuda()
    else:
        return Variable(x, requires_grad=requires_grad)

def to_device(src, ref=None, var=True):
    if type(src) is list:
        src = torch.Tensor(src)
    src = to_variable(src) if var else src
    return src.cuda(ref.get_device()) if (ref is not None and ref.is_cuda) else src


def one_hot(nwords, ref=None):
    '''
    :param nwords: 
    :param ref: 
    :return: one-hot encoded tensor.
    '''
    eye_tensor = to_device(torch.eye(nwords), ref, var=False)
    def f(tensor_int):
        if type(tensor_int) is int:
            tensor_int = [tensor_int]
        if type(tensor_int) is list or np.ndarray:
            tensor_int = torch.LongTensor(tensor_int)
        
        tensor_int = to_device(tensor_int, ref, var=False)
        size = list(tensor_int.size())
        flat = tensor_int.view(-1)
        flat_emb = torch.index_select(eye_tensor, 0, flat)
        new_shape = size + [nwords]
        emb = flat_emb.view(*new_shape)
        return Variable(emb)

    return f


def multinomial(pred):
    shape = pred.size()
    valid_shape = list(pred.size())
    valid_shape[-1] = 1
    mat_action = pred.view(-1,shape[-1]).multinomial()
    return mat_action.view(*valid_shape)

def creteria_(pred, label):
    # both of them should be Tensor (N, dim)
    _, target = label.topk(1, dim=1)
    loss = F.nll_loss(F.log_softmax(pred), target[:,0])
    return loss

def validate(model, valid_flow, cuda=False):
    acc = []
    pred_list, target_list = [], []
    for data, label, _ in valid_flow:
        data, label = to_variable(data, cuda=cuda), to_variable(label, cuda=cuda)
        pred = model.forward(data)
        _, target = label.topk(1,dim=1)
        pred_list.append(pred)
        target_list.append(target)
    
    acc = cls_accuracy( torch.cat(pred_list,0), torch.cat(target_list,0) )
    return acc[0].data.cpu().numpy()

def cls_accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    Parameters:
    -----------
    output: (N, dim) torch tensor
    
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size) )
    return res


def reduce_sum(inputs, dim=None, keep_dim=False):
    if dim is None:
        return torch.sum(inputs)
    output = torch.sum(inputs, dim)
    if not keep_dim:
        return output
    else:
        return expand_dims(output, dim)
        
    
def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)
    can also be performed on batch of vectors.
    Parameters:
    ----------
    u, v: Tensor (m,) or (b,m)

    Returns: 
    ---------
    Tensor (m, n) or (b, m, n)
    
    """
    u_shape = u.size()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) > 2 and is_batch:
        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

    if v is None:
        v = u
    v_shape = v.size()

    m = u_shape[0] if not is_batch else u_shape[1]
    n = v_shape[0] if not is_batch else v_shape[1]
    
    u = expand_dims(u, axis=-1)
    new_u_shape = list(u.size())
    new_u_shape[-1] = n
    U_ = u.expand(*new_u_shape)

    v = expand_dims(v, axis=-2)
    new_v_shape = list(v.size())
    new_v_shape[-2] = m
    V_ = v.expand(*new_v_shape)

    return U_ + V_


def cumprod(inputs, dim = 1, exclusive=True):
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.cumprod.md

    if type(inputs) is not Variable:
        temp = torch.cumprod(inputs, dim)
        if not exclusive:
            return temp
        else:
            temp =  temp / (input[0].expand_dims(0).expand_as(temp) + 1e-8)
            temp[-1] = temp[-1]/(input[-1]+1e-8)
            return temp
    else:
        shape_ = inputs.size()
        ndim = len(shape_)
        n_slot = shape_[dim]
        output = Variable(inputs.data.new(*shape_).fill_(1.0), requires_grad = True)
        slice_ = [slice(0,None,1) for _ in range(ndim)]
        results = [[]] * n_slot
            
        for ind in range(0, n_slot):   
            this_slice, last_slice = copy(slice_), copy(slice_)
            this_slice[dim] = ind
            last_slice[dim] = ind-1      
            this_slice = tuple(this_slice)
            last_slice = tuple(last_slice)
            if exclusive: 
                if ind > 0:   
                    results[ind]  = results[ind-1]*inputs[last_slice]
                else:
                    results[ind] =  torch.div(inputs[this_slice], inputs[this_slice]+1e-8)
            else:    
                if ind > 0:   
                    results[ind]  = results[ind - 1]*inputs[this_slice]
                else:
                    results[ind] =  inputs[this_slice]
        
        return torch.stack(results, dim)

            
def expand_dims(input, axis=0):
    input_shape = list(input.size())
    if axis < 0:
        axis = len(input_shape) + axis + 1
    input_shape.insert(axis, 1)
    return input.view(*input_shape)

def softmax(input, axis=1):
    """ 
    Apply softmax on input at certain axis.
    
    Parammeters:
    ----------
    input: Tensor (N*L or rank>2)
    axis: the axis to apply softmax
    
    Returns: Tensor with softmax applied on that dimension.
    """
    
    input_size = input.size()
    
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    
    soft_max_nd = soft_max_2d.view(*trans_size)
    
    return soft_max_nd.transpose(axis, len(input_size)-1)

def get_content_address(memory_matrix, query_keys, strengths):
    """
    retrives a content-based adderssing weights given the keys

    Parameters:
    ----------
    memory_matrix: Tensor (batch_size, mem_slot, mem_size)
        the memory matrix to lookup in
    query_keys: Tensor (batch_size, mem_size, number_of_keys)
        the keys to query the memory with
    strengths: Tensor (batch_size, number_of_keys, )
        the list of strengths for each lookup query_keys

    Returns: Tensor (batch_size, mem_slot, number_of_keys)
        The list of lookup weightings for each provided key
    """
    # cos_dist is (batch_size, mem_slot, number_of_keys)
    cos_dist = cosine_distance(memory_matrix, query_keys)

    strengths = expand_dims(strengths, 1).expand_as(cos_dist)
    # apply_dict(locals())
    return softmax(cos_dist * strengths, 1)


def update_read_vectors(memory_matrix, read_weights):
    """
    reads, updates, and returns the read vectors of the recently updated memory

    Parameters:
    ----------
    memory_matrix: Tensor (batch_size, mem_slot, mem_size)
        the recently updated memory matrix
    read_weights: Tensor (batch_size, mem_slot, read_heads)
        the amount of info to read from each memory location by each read head

    Returns: Tensor (batch_size, mem_size, read_heads)
    """

    updated_read_vectors = torch.bmm(memory_matrix.transpose(1, 2), read_weights)

    # apply_dict(locals())

    return updated_read_vectors

def cosine_similarity(first, second):
    """
    compute the cosine similarity between keys to each of the 
    memory slot.

    Parameters:
    ----------
    first: Tensor (batch_size, dim)
        the memory matrix to lookup in
    second: Tensor (batch_size, dim)
        the keys to query the memory with
    Returns: Tensor (batch_size, 1)
    """
    first_norm  = torch.norm(first,  2, 1)
    second_norm = torch.norm(second, 2, 1)

    normalized_first  = torch.div(first,  first_norm.expand_as(first) + 1e-10)
    normalized_second = torch.div(second, second_norm.expand_as(second) + 1e-10)

    return torch.sum(normalized_first*normalized_second,1)

def cosine_distance(memory_matrix, keys):
    """
    compute the cosine similarity between keys to each of the 
    memory slot.

    Parameters:
    ----------
    memory_matrix: Tensor (batch_size, mem_slot, mem_size)
        the memory matrix to lookup in
    keys: Tensor (batch_size, mem_size, number_of_keys)
        the keys to query the memory with
    strengths: Tensor (batch_size, number_of_keys, )
        the list of strengths for each lookup key

    Returns: Tensor (batch_size, mem_slot, number_of_keys)
        The list of lookup weightings for each provided key
    """
    memory_norm = torch.norm(memory_matrix, 2, 2)
    keys_norm = torch.norm(keys, 2, 1)

    normalized_mem = torch.div(memory_matrix, memory_norm.expand_as(memory_matrix) + 1e-9)
    normalized_keys = torch.div(keys, keys_norm.expand_as(keys) + 1e-9)

    return torch.bmm(normalized_mem, normalized_keys)


def display_loss(steps, values, plot=None, name='default', legend= None):
    if plot is None:
        plot = Visdom()
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    repeat_steps = [steps]*n_lines
    steps  = np.array(repeat_steps).transpose()
    values = np.array(values).transpose()
    win = name + '_loss'
    res = plot.line(
            X = steps,
            Y=  values,
            win= win,
            update='append',
            opts=dict(title = win, legend=legend)
        )
    if res != win:
        plot.line(
            X = steps,
            Y=  values,
            win=win,
            opts=dict(title = win,legend=legend)
        )

