import util.io as io

import os
import sys
import numpy as np
import time
from PIL import Image

import torch.utils.data as data
from torchvision import transforms


## helper function ##

class StandardFaceTransform(object):
    '''
    Standard transformation for face image. 

    The input image should be aligned and cropped to 178 * 218. The output is 3x224x224 Tensor
    to serve as input of VGG or ResNet

    Args:
        crop_size (int)
        y_offset (int)
        flip (bool): perform random horizontal flippings
    '''

    def __init__(self, crop_size = 128, y_offset = 15, flip = False):

        self.crop_size = crop_size
        self.y_offset = y_offset
        self.flip = flip

        if self.flip:
            self.post_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Scale(size = 224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.post_transform = transforms.Compose([
                transforms.Scale(size = 224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __call__(self, img):

        # standardize image size
        w, h = 178, 218 # standard image size
        if img.size != (w, h):
            img.resize((w, h), Image.BILINEAR)

        # crop face
        x1 = int(round((w - crop_size) / 2.0))
        y1 = int(round((h - crop_size) / 2.0) + y_offset)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # post transform
        img = self.post_transform(img)

        return img


def load_dataset(dset_name, subset = 'train'):

    if dset_name == 'imdb_wiki':

        img_root = './datasets/IMDB-WIKI/Images'
        sample_lst_fn = './datasets/IMDB-WIKI/Annotations/imdb_wiki_%s.json' % subset

        if subset == 'train':
            transform = StandardFaceTransform(flip = True)
        else:
            transform = StandardFaceTransform(flip = False)

        return Image_Age_Dataset(img_root = img_root, sample_lst_fn = sample_lst_fn, age_std = False,
            age_rng = None, transform = transform)

    else:
        raise Exception('Unknown dataset "%s"' % dset_name)


class Image_Age_Dataset(data.Dataset):
    '''
    Pytorch Wrapper for aging datasets
    '''

    def __init__(self, img_root, sample_lst_fn, age_std = False, age_dist = False, age_rng = None, transform = None):
        '''
            img_root    : root path of image files
            sample_lst_fn  : a json file containing an image list. Each element should be a dict with keys: "age", "image", "identity", "person_id"
            age_std     : load age standard deviation. The sample should have the field "std"
            age_dist    : load age probability distribution. The sample should have the field "dist" 
            age_rng     : a tuple (min_age, max_age). Samples out of the range will be ignored.

        '''

        self.img_root = img_root
        self.age_std = age_std
        self.age_dist = age_dist
        self.age_rng = age_rng
        self.transform = transform

        self.sample_lst = io.load_json(sample_lst_fn)

        if self.age_rng:
            self.sample_lst = [s for s in self.sample_lst if s['age'] >= age_rng[0] and s['age'] <= age_rng[1]]
        
        if self.age_std:
            assert 'std' in self.sample_lst[0], 'fail to load age_std information from %s' % sample_lst_fn

        if self.age_dist:
            assert 'dist' in self.sample_lst[0], 'fail to load age_dist informaiton from %s' % sample_lst_fn


    def __len__(self):

        return len(self.sample_lst)

    def __getitem__(self, index):
        '''
        output data format:

            img: Tensor, containing transformed image
            age: float value
            std: float value
        '''

        s = self.sample_lst[index]

        img = Image.open(os.path.join(self.img_root, s['image'])).convert('RGB')

        if self.transform:
            img = self.transform(img)

        age = s['age']

        if self.age_std:
            std = s['std']
        else:
            std = 0

        if self.age_dist:
            dist = s['dist']
        else:
            dist = 0

        return img, age, (std, dist)


    def set_age_range(self, age_rng):

        assert len(age_rng) == 2

        

        new_len = len(self.sample_lst)



