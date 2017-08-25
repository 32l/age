import util.io as io

import os
import sys
import numpy as np
import time
from PIL import Image

import torch.utils.data as data
from torchvision import transforms


## helper function ##

def load_dataset(dset_name):

    if dset_name == 'imdb_wiki':

        img_root = './datasets/IMDB-WIKI/Images'
        sample_lst_fn = './datasets/IMDB-WIKI/Annotations/imdb_wiki.json'

        transform = transforms.Compose([
            transforms.Scale(size = 256),
            transforms.CenterCrop(size = 224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])

        return Image_Age_Dataset(img_root = img_root, sample_lst_fn = sample_lst_fn, age_std = False,
            age_rng = None, transform = transform)

    else:
        raise Exception('Unknown dataset "%s"' % dset_name)


class Image_Age_Dataset(data.Dataset):
    '''
    Pytorch Wrapper for aging datasets
    '''

    def __init__(self, img_root, sample_lst_fn, age_std = False, age_rng = None, transform = None):
        '''
            img_root    : root path of image files
            sample_lst_fn  : a json file containing an image list. Each element should be a dict with keys: "age", "image", "identity", "person_id"
            age_std     : load age standard deviation. The image image element should have the field "age_std"
            age_rng     : a tuple (min_age, max_age). Samples out of the range will be ignored.

        '''

        self.img_root = img_root
        self.age_std = age_std
        self.age_rng = age_rng
        self.transform = transform

        self.sample_lst = io.load_json(sample_lst_fn)

        if self.age_rng:
            self.set_age_range()


        if self.age_std:
            assert 'age_std' in self.sample_lst[0], 'fail to load age_std information from %s' % sample_lst_fn

        

        


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
            std = s['age_std']
        else:
            std = 0

        return img, age, std


    def set_age_range(self, age_rng):

        assert len(age_rng) == 2

        old_len = len(self.sample_lst)

        self.sample_lst = [s for s in self.sample_lst if s['age'] >= age_rng[0] and s['age'] <= age_rng[1]]
        self.age_rng = age_rng

        new_len = len(self.sample_lst)

