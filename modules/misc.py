from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

from PIL import Image, ImageOps
import numpy as np


class Smooth_Loss():
    '''
    wrapper of pytorch loss layer.
    '''

    def __init__(self, crit):
        self.crit = crit
        self.clear()

    def __call__(self, out, gt, *extra_input):
        loss = self.crit(out, gt, *extra_input)
        self.weight_buffer.append(out.size(0))

        if isinstance(loss, Variable):
            self.buffer.append(loss.data[0])
        elif isinstance(loss, torch.tensor._TensorBase):
            self.buffer.append(loss[0])
        else:
            self.buffer.append(loss)

        return loss

    def clear(self):
        self.buffer = []
        self.weight_buffer = []

    def smooth_loss(self, clear = False):
        loss = sum([l * w for l, w in zip(self.buffer, self.weight_buffer)]) / sum(self.weight_buffer)
        if clear:
            self.clear()
        return loss


class Video_Loss():
    '''
    wrapper to apply image-level loss function to video data.
    '''

    def __init__(self, crit, same_sz = False):
        self.crit = crit
        self.same_sz = same_sz

    def __call__(self, out, gt, seq_len):
        '''
        input:
            out: [bsz, max_seq_len, *output_sz]
            gt : [bsz, *gt_sz]. if same_sz == Ture, gt has the same size as "out"
            seq_len: [bsz, 1]
        '''

        #### unfolding method 1 ####
        out_unfold = []
        gt_unfold = []

        gt_sz = gt.size()[1::]

        for i, l in enumerate(seq_len):
            l = int(l.data[0])

            out_unfold.append(out[i, 0:l])

            if not self.same_sz:
                gt_unfold.append(gt[i:(i+1)].expand(l, *gt_sz))
            else:
                gt_unfold.append(gt[i,0:l])

        out_unfold = torch.cat(out_unfold)
        gt_unfold = torch.cat(gt_unfold)


        #### unfolding method 2 ####
        # bsz, max_seq_len = out.size()[0:2]
        # output_sz = out.size()[2::]
        # gt_sz = gt.size()[1::]

        # out = out.view(bsz * max_seq_len, *output_sz)
        # gt = gt.view(bsz, 1, *gt_sz).expand(bsz, max_seq_len, *gt_sz).view(bsz * max_seq_len, *gt_sz)
        # index = torch.LongTensor([i for i in xrange(bsz * max_seq_len) if seq_len.data[i//max_seq_len][0] > (i%max_seq_len)])
        # out_unfold = torch.index_select(out, 0, index)
        # gt_unfold = torch.index_select(gt, 0, index)

        return self.crit(out_unfold, gt_unfold)


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
        self.std_buffer = np.array([])
        
    def add(self, age_out, age_gt, age_std = None):

        inputs = [age_out, age_gt, age_std]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32).flatten()

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32).flatten()

            elif isinstance(inputs[i], torch.tensor._TensorBase):
                inputs[i] = inputs[i].numpy().astype(np.float32).flatten()

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32).flatten()

        age_out, age_gt, age_std = inputs
        assert age_out.size == age_gt.size
        self.err_buffer = np.concatenate((self.err_buffer, age_out - age_gt))

        if age_std is not None:
            self.std_buffer = np.concatenate((self.std_buffer, age_std))

    def ca(self, k = 10):

        if self.err_buffer.size == 0:
            raise Exception('buffer empty')
        else:
            return (np.abs(self.err_buffer).round() <= k).sum() / self.err_buffer.size * 100.

    def mae(self):

        if self.err_buffer.size == 0:
            raise Exception('buffer empty')
        else:
            return np.abs(self.err_buffer).mean()

    def lap_err(self):

        if self.err_buffer.size == 0:
            raise Exception('buffer empty')
        elif self.std_buffer.size == 0:
            return -1
        else:
            return (1.-np.exp(-np.power(self.err_buffer/self.std_buffer, 2)/2)).mean()

class Video_Age_Analysis():
    '''
    anylize video age estimatino performance
    '''

    def __init__(self):
        self.clear()

    def clear(self):
        self.age_out_buffer = None
        self.age_gt_buffer = None
        self.std_buffer = None
        self.seq_len_buffer = None
        self.mask = None

    def add(self, age_out, age_gt, seq_len, age_std = None):
        
        inputs = [age_out, age_gt, seq_len, age_std]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32)

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32)

            elif isinstance(inputs[i], torch.tensor._TensorBase):
                inputs[i] = inputs[i].numpy().astype(np.float32)

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32)

        age_out, age_gt, seq_len, age_std = inputs

        age_gt = age_gt.flatten()
        age_std = age_std.flatten()
        seq_len = seq_len.flatten().astype(np.int)

        if self.age_out_buffer is None:
            self.age_out_buffer = age_out
            self.age_gt_buffer = age_gt
            self.seq_len_buffer = seq_len
            if age_std is not None:
                self.std_buffer = age_std
        else:
            self.age_out_buffer = np.concatenate((self.age_out_buffer, age_out))
            self.age_gt_buffer = np.concatenate((self.age_gt_buffer, age_gt))
            self.seq_len_buffer = np.concatenate((self.seq_len_buffer, seq_len))
            if age_std is not None:
                self.std_buffer = np.concatenate((self.std_buffer, age_std))

        self.mask = None


    def get_mask(self):
        mask = self.age_out_buffer.copy()
        for i in xrange(mask.shape[0]):
            mask[i, 0:self.seq_len_buffer[i]] = self.age_gt_buffer[i]

        self.mask = mask

    def mae(self):

        if self.mask is None:
            self.get_mask()

        mae = np.abs(self.age_out_buffer - self.mask).sum() / self.seq_len_buffer.sum()

        return mae

    def ca(self, k = 10):

        if self.mask is None:
            self.get_mask()

        n = (np.abs(self.age_out_buffer - self.mask).round() <= k).sum() + self.seq_len_buffer.sum() - self.mask.size

        return n / self.seq_len_buffer.sum() * 100.

    def lap_err(self):

        if self.std_buffer is None:
            return -1.
        else:
            if self.mask is None:
                self.get_mask()

            err_mask = self.age_out_buffer - self.mask
            std_mask = self.std_buffer.reshape(-1, 1).repeat(err_mask.shape[1], axis = 1)

            return (1. - np.exp(-np.power(err_mask/std_mask, 2)/2)).sum() / self.seq_len_buffer.sum()

    def stable_der(self):
        '''
        stability measured by 1-order derivative
        '''

        der_mask = np.abs(self.age_out_buffer[:,0:-1] - self.age_out_buffer[:, 1::])
        der_sum = 0

        for i, l in enumerate(self.seq_len_buffer):
            l = int(l)
            der_sum += der_mask[i, 0:(l-1)].sum()

        return der_sum / (self.seq_len_buffer - 1).sum()

    def stable_range(self):
        '''
        stability measured by the variation range (max-min) in a video clip
        '''

        rng_sum = 0
        for i, l in enumerate(self.seq_len_buffer):
            l = int(l)
            age_seq = self.age_out_buffer[i, 0:l]
            rng_sum += age_seq.max() - age_seq.min()

        return rng_sum / self.seq_len_buffer.size


class MeanAP():
    '''
    compute meanAP
    '''

    def __init__(self):
        self.clear()

    def clear(self):
        self.score = None
        self.label = None

    def add(self, new_score, new_label):

        inputs = [new_score, new_label]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32)

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32)

            elif isinstance(inputs[i], torch.tensor._TensorBase):
                inputs[i] = inputs[i].numpy().astype(np.float32)

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32)

        new_score, new_label = inputs
        assert new_score.shape == new_label.shape, 'shape mismatch: %s vs. %s' % (new_score.shape, new_label.shape)

        self.score = np.concatenate((self.score, new_score), axis = 0) if self.score is not None else new_score
        self.label = np.concatenate((self.label, new_label), axis = 0) if self.label is not None else new_label


    def compute_mean_ap(self):

        score, label = self.score, self.label

        assert score is not None and label is not None
        assert score.shape == label.shape, 'shape mismatch: %s vs. %s' % (score.shape, label.shape)
        assert(score.ndim == 2)
        M, N = score.shape[0], score.shape[1]

        # compute tp: column n in tp is the n-th class label in descending order of the sample score.
        index = np.argsort(score, axis = 0)[::-1, :]
        tp = label.copy().astype(np.float)
        for i in xrange(N):
            tp[:, i] = tp[index[:,i], i]
        tp = tp.cumsum(axis = 0)

        m_grid, n_grid = np.meshgrid(range(M), range(N), indexing = 'ij')
        tp_add_fp = m_grid + 1    
        num_truths = np.sum(label, axis = 0)
        # compute recall and precise
        rec = tp / num_truths
        prec = tp / tp_add_fp

        prec = np.append(np.zeros((1,N), dtype = np.float), prec, axis = 0)
        for i in xrange(M-1, -1, -1):
            prec[i, :] = np.max(prec[i:i+2, :], axis = 0)
        rec_1 = np.append(np.zeros((1,N), dtype = np.float), rec, axis = 0)
        rec_2 = np.append(rec, np.ones((1,N), dtype = np.float), axis = 0)
        AP = np.sum(prec * (rec_2 - rec_1), axis = 0)
        AP[np.isnan(AP)] = -1 # avoid error caused by classes that have no positive sample

        assert((AP <= 1).all())

        AP = AP * 100.
        meanAP = AP[AP != -1].mean()

        return meanAP, AP

    def compute_mean_ap_pn(self):
        '''
        compute the average of true-positive-rate and true-negative-rate
        '''

        score, label = self.score, self.label

        assert score is not None and label is not None
        assert score.shape == label.shape, 'shape mismatch: %s vs. %s' % (score.shape, label.shape)
        assert(score.ndim == 2)

        # compute true-positive and true-negative
        tp = np.where(np.logical_and(score > 0.5, label == 1), 1, 0)
        tn = np.where(np.logical_and(score < 0.5, label == 0), 1, 0)

        # compute average precise
        p_pos = tp.sum(axis = 0) / (label == 1).sum(axis = 0)
        p_neg = tn.sum(axis = 0) / (label == 0).sum(axis = 0)

        ave_p = (p_pos + p_neg) / 2

        ave_p = ave_p * 100.
        ave_ave_p = ave_p.mean()

        return ave_ave_p, ave_p
            
class Pose_MAE():

    def __init__(self, pose_dim):
        self.pose_dim = pose_dim
        self.clear()

    def clear(self):
        self.err_buffer = None

    def add(self, pose_out, pose_gt):
        inputs = [pose_out, pose_gt]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32)

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32)

            elif isinstance(inputs[i], torch.tensor._TensorBase):
                inputs[i] = inputs[i].numpy().astype(np.float32)

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32)

        pose_out, pose_gt = inputs
        assert pose_out.size == pose_gt.size

        if self.err_buffer is None:
            self.err_buffer = pose_out - pose_gt
        else:
            self.err_buffer = np.concatenate((self.err_buffer, pose_out - pose_gt))

    def mae(self):
        assert self.err_buffer is not None
        return np.abs(self.err_buffer).mean(axis = 0) / np.pi * 180.

class L2NormLoss(nn.Module):

    def forward(self, out_1, out_2):
        assert out_1.is_same_size(out_2)

        bsz = out_1.size(0)

        return (out_1 - out_2).norm(p = 2) / bsz


class BlankLoss(nn.Module):

    def forward(self, out_1, out_2 = None):
        return out_1.sum() / out_1.size(0)

class BCEAccuracy(nn.Module):

    def forward(self, out, label):

        if isinstance(out, Variable):
            out = out.data
        if isinstance(label, Variable):
            label = label.data

        return (((out >= 0.5) & (label >= 0.5)) | ((out < 0.5) & (label < 0.5))).float().sum() / label.numel()


