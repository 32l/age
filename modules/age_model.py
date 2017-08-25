# Age Estimation Model

from __future__ import division, print_function

import util.io as io
import dataset
import misc
import opt_parser

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import numpy as np
from collections import OrderedDict
import time


class AgeModel(nn.Module):

    '''
    basic age model
    '''

    def __init__(self, opts = None, fn = None):

        assert (opts or fn), 'Error: either "opts" or "fn" should be provided'

        super(AgeModel, self).__init__()


        ## set model opts

        if fn:
            opts = torch.load(fn)['opts']
        
        self.opts = opts

        ## create model

        