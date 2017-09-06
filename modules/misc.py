from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

from PIL import Image, ImageOps
import numpy as np

class Loss_Buffer():
    '''
    average loss
    '''
    def __init__(self, size):

        assert size > 0
        self.size = size
        self.buffer = []
        self.weight_buffer = []

    def __call__(self, new_loss = None, new_weight = 1.0):
        
        if new_loss is not None:
            if len(self.buffer) == self.size:
                self.buffer.pop(0)
                self.weight_buffer.pop(0)
            
            self.buffer.append(new_loss)
            self.weight_buffer.append(new_weight)

        output = sum([l * w for l, w in zip(self.buffer, self.weight_buffer)]) / sum(self.weight_buffer)
        
        return output


class Ordinal_Hyperplane_Loss(nn.Module):

    def __init__(self, relaxation = 3, ignore_index = -1, sample_normalize = 1):
        super(Ordinal_Hyperplane_Loss, self).__init__()
        self.rlx = relaxation
        self.ignore_index = ignore_index
        self.sample_normalize = sample_normalize

    def forward(self, fc_out, label):
        '''
        Input:
            fc_out: (batch_size, num_age), torch.Tensor
            label: (batch_size, 1), torch.LongTensor
        '''
        bsz = fc_out.size(0)
        asz = fc_out.size(1)


        # label_grid: [[0, 0, ..., 0], [1, 1, ..., 1], ...]
        _, label_grid = np.meshgrid(range(bsz), range(asz), indexing = 'ij')

        # label_exp: [[a0, a0, ..., a0], [a1, a1, ..., a1], ...]
        label_exp = label.view(bsz, 1).expand(bsz, asz)
        label_exp_np = label_exp.data.cpu().numpy()

        # label_ordinal
        label_ordinal = np.where(label_exp_np > label_grid, 1, 0)
        label_ordinal = Variable(fc_out.data.new(label_ordinal))

        # relaxation_mask
        rlx_mask = np.where(np.abs(label_exp_np - label_grid) < self.rlx, 0, 1)
        rlx_mask = Variable(fc_out.data.new(rlx_mask))

        # valid
        valid_mask = label_exp >= 0

        # loss
        if self.sample_normalize == 1:
            loss = ((fc_out - label_ordinal).pow(2) * rlx_mask)[valid_mask].sum() / bsz
        else:
            loss = ((fc_out - label_ordinal).pow(2) * rlx_mask)[valid_mask].sum() / valid_mask.data.sum()

        return loss



class Mean_Absolute_Error():
    '''
    compute mean absolute error
    '''
    def __init__(self, ignore_index = -1):
        self.ignore_index = ignore_index

    def __call__(self, age_out, age_gt):

        # valid_mask = (age_gt != self.ignore_index).data.cpu()
        valid_mask = age_gt != self.ignore_index

        mae = (age_out - age_gt)[valid_mask].data.abs().mean()

        return mae


class Cumulative_Accuracy():
    '''
    compute cumulative accuracy
    '''
    def __init__(self):

        self.clear()

    def clear(self):

        self.err_buffer = np.array([])
        
    def add(self, age_out, age_gt):

        inputs = [age_out, age_gt]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32).flatten()

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32).flatten()

            elif isinstance(inputs[i], torch.tensor._TensorBase):
                inputs[i] = inputs[i].numpy().astype(np.float32).flatten()

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32).flatten()

        age_out, age_gt = inputs
        assert age_out.size == age_gt.size
        self.err_buffer = np.concatenate((self.err_buffer, age_out - age_gt))

    def ca(self, k = 10):

        if self.err_buffer.size == 0:
            return np.nan
        else:
            return (np.abs(self.err_buffer) < k).sum() / self.err_buffer.size * 100.

    def mae(self):

        if self.err_buffer.size == 0:
            return np.nan
        else:
            return np.abs(self.err_buffer).mean()


