import torch
import torch.utils.data as data
from torchvision import transforms

import util.io as io

import os
import sys
import collections
import numpy as np
import time
from PIL import Image


## helper function ##
def load_age_dataset(dset_name, subset = 'train', alignment = 'none', crop_size = 128, **argv):

    if dset_name == 'imdb_wiki' or dset_name == 'imdb_wiki_good':

        if dset_name == 'imdb_wiki':
            sample_lst_fn = './datasets/IMDB-WIKI/Annotations/imdb_wiki_%s.json' % subset
        else:
            sample_lst_fn = './datasets/IMDB-WIKI/Annotations/imdb_wiki_good_%s.json' % subset

        if alignment == '3':
            img_root = './datasets/IMDB-WIKI/Images_aligned_3'
        elif alignment == '21':
            img_root = './datasets/IMDB-WIKI/Images_aligned_21'
        elif alignment == 'none':
            img_root = './datasets/IMDB-WIKI/Images'
        else:
            raise Exception('Invalid alignment mode %s for %s' % (alignment, dset_name))

        if subset == 'train':
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        else:
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)

    elif dset_name == 'megaage':

        sample_lst_fn = './datasets/megaAge/Label/megaage_%s.json' % subset

        if alignment == '21':
            img_root = './datasets/megaAge/'
        else:
            raise Exception('Invalid alignment mode %s for %s' % (alignment, dset_name))

        if subset == 'train':
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        else:
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)

    elif dset_name == 'morph':
        sample_lst_fn = './datasets/morph/Label/morph_%s.json' % subset
        img_root = './datasets/morph/sy'

        if subset == 'train':
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        else:
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)

    elif dset_name == 'lap':

        argv['age_std'] = True
        
        if subset == 'train':
            sample_lst_fn = './datasets/LAP_2016/Label/lap_trainval.json'
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        elif subset == 'test':
            sample_lst_fn = './datasets/LAP_2016/Label/lap_test.json'
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)

        if alignment == '3':
            img_root = './datasets/LAP_2016/Image_aligned_3'
        elif alignment == '21':
            img_root = './datasets/LAP_2016/Image_aligned_21'
        elif alignment == 'none':
            img_root = './datasets/LAP_2016/Image'
        else:
            raise Exception('Invalid alignment mode %s for %s' % (alignment, dset_name))
    
    elif dset_name == 'video_age':

        argv['age_std'] = True

        sample_lst_fn = './datasets/video_age/Labels/v2.0_image_%s.json' % subset
        if subset.startswith('train'):
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        elif subset == 'test':
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)

        img_root = './'

    else:
        raise Exception('Unknown dataset "%s"' % dset_name)


    return Image_Age_Dataset(img_root = img_root, sample_lst_fn = sample_lst_fn, 
            transform = transform, **argv)


def load_video_age_dataset(version = '2.0', subset = 'test', split = '', crop_size = 128, **argv):

    if version == '1.0':
        age_fn = 'datasets/video_age/Labels/v1.0_age.json'
        video_fn = 'datasets/video_age/Labels/v1.0_video.json'
        split_fn = 'datasets/video_age/Labels/v1.0_split.json'
    if version == '2.0':
        age_fn = 'datasets/video_age/Labels/v2.0_age.json'
        video_fn = 'datasets/video_age/Labels/v2.0_video.json'
        split_fn = 'datasets/video_age/Labels/v2.0_split.json'
    else:
        raise Exception('Invalid version of video_age dataset: %s' % version)

    if subset == 'train':
        if split != '':
            subset = 'train_' + split
        flip = True
    else:
        flip = False
            
    transform = StandardFaceTransform(flip = False, crop_size = crop_size)

    return Video_Age_Dataset(subset = subset, age_fn = age_fn, video_fn = video_fn, 
        split_fn = split_fn, transform = transform, flip = flip, **argv)



def load_pose_dataset(dset_name, subset = 'train', alignment = 'none', crop_size = 128, debug = 0):

    if dset_name == 'aflw':
        sample_lst_fn = './datasets/AFLW/Label/aflw_%s.json' % subset
        if alignment == 'none':
            img_root = './datasets/AFLW/flickr'
        else:
            img_root = './datasets/AFLW/Image_aligned'

        if subset == 'train':
            flip = True
        else:
            flip = False

        transform = StandardFaceTransform(flip = False, crop_size = crop_size)

    else:
        raise Exception('Unknown dataset "%s"' % dset_name)

    return Pose_Dataset(img_root = img_root, sample_lst_fn = sample_lst_fn, 
        transform = transform, flip = flip, debug = debug)

def load_attribute_dataset(dset_name, subset = 'train', alignment = 'none', crop_size = 128, debug = 0):
    if dset_name == 'celeba':
        sample_lst_fn = './datasets/CelebA/Label/celeba_%s.json' % subset
        img_root = './datasets/CelebA/Image_aligned'
        attr_name_fn = './datasets/CelebA/Label/attr_name_lst.txt'

        if subset == 'train':
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        else:
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)
    elif dset_name == 'celeba_selc1':
        sample_lst_fn = './datasets/CelebA/Label/celeba_selc1_%s.json' % subset
        img_root = './datasets/CelebA/Image_aligned'
        attr_name_fn = './datasets/CelebA/Label/attr_name_selc1_lst.txt'

        if subset == 'train':
            transform = StandardFaceTransform(flip = True, crop_size = crop_size)
        else:
            transform = StandardFaceTransform(flip = False, crop_size = crop_size)

    else:
        raise Exception('Unknwon dataset %s'%dset_name)

    return Attribute_Dataset(img_root = img_root, sample_lst_fn = sample_lst_fn, attr_name_fn = attr_name_fn,
        transform = transform, debug = debug)

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
        x1 = int(round((w - self.crop_size) / 2.0))
        y1 = int(round((h - self.crop_size) / 2.0) + self.y_offset)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # post transform
        img = self.post_transform(img)

        return img


class Image_Age_Dataset(data.Dataset):
    '''
    Pytorch Wrapper for aging datasets
    '''

    def __init__(self, img_root, sample_lst_fn, age_std = False, age_dist = False, 
        age_rng = None, transform = None, debug = 0):
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
        self.debug = debug

        self.sample_lst = io.load_json(sample_lst_fn)

        if self.age_rng:
            self.sample_lst = [s for s in self.sample_lst if s['age'] >= self.age_rng[0] and s['age'] <= self.age_rng[1]]
        
        if self.age_std:
            assert 'std' in self.sample_lst[0], 'fail to load age_std information from %s' % sample_lst_fn

        if self.age_dist:
            assert 'dist' in self.sample_lst[0], 'fail to load age_dist informaiton from %s' % sample_lst_fn


        # standardize data format
        # for i in xrange(len(self.sample_lst)):
        #     self.sample_lst[i]['age'] = float(self.sample_lst[i]['age'])

        self.len = len(self.sample_lst)

        print('[Image_Age_Dataset] Sample: %d   Age Range: %s\n\tSample List: %s\n\tImage Root: %s' %
            (self.len, self.age_rng, sample_lst_fn, self.img_root))


    def __len__(self):

        if self.debug == 1:
            return 1
        else:
            return self.len

    def __getitem__(self, index):
        '''
        output data format:

            img: Tensor, containing transformed image
            age: float value
            std: float value
        '''
        if self.debug == 1:
            s = self.sample_lst[0]
        else:
            s = self.sample_lst[index]

        img = Image.open(os.path.join(self.img_root, s['image'])).convert('RGB')
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

class Pose_Dataset(data.Dataset):

    def __init__(self, img_root, sample_lst_fn, transform = None, flip = False, debug = 0):

        self.img_root = img_root
        self.sample_lst = io.load_json(sample_lst_fn)
        self.transform = transform
        self.flip = flip
        self.debug = debug

        self.len = len(self.sample_lst)
        print('[Pose_Dataset] Sample: %d  Sample List: %s\n\tImage Root: %s' %
            (self.len, sample_lst_fn, self.img_root))


    def __len__(self):

        if self.debug == 1:
            return 1
        else:
            return self.len

    def __getitem__(self, index):
        '''
        output data format:

        img: Tensor, size = (3, 224, 224)
        pose: Tensor, size = (3,), [yaw, pitch, roll]
        '''

        if self.debug == 1:
            s = self.sample_lst[0]
        else:
            s = self.sample_lst[index]

        img = Image.open(os.path.join(self.img_root, s['image'])).convert('RGB')
        pose = np.array(s['pose'])
        
        if self.flip and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            pose[0] = -pose[0] # yaw need to be reversed

        img = self.transform(img)

        return img, pose

class Attribute_Dataset(data.Dataset):

    def __init__(self, img_root, sample_lst_fn, attr_name_fn, transform = None, debug = 0):

        self.img_root = img_root
        self.transform = transform
        self.debug = debug

        self.sample_lst = io.load_json(sample_lst_fn)
        self.len = len(self.sample_lst)

        self.attr_name_lst = io.load_str_list(attr_name_fn)
        assert len(self.attr_name_lst) == len(self.sample_lst[0]['attr'])

        print('[Attribute_Dataset] Sample: %d  Sample List: %s\n\tImage Root: %s' %
            (self.len, sample_lst_fn, self.img_root))


    def __len__(self):

        if self.debug == 1:
            return 1
        else:
            return self.len

    def __getitem__(self, index):
        '''
        output data format:

        img: Tensor, size = (3, 224, 224)
        attr: Tensor, size = (3, num_attr), [yaw, pitch, roll]
        '''

        if self.debug == 1:
            s = self.sample_lst[0]
        else:
            s = self.sample_lst[index]

        img = Image.open(os.path.join(self.img_root, s['image'])).convert('RGB')
        attr = np.array(s['attr'])
                
        img = self.transform(img)

        return img, attr

class Video_Age_Dataset(data.Dataset):
    def __init__(self, subset, age_fn, video_fn, split_fn, mode = 'video', max_len = 17, age_rng = None, transform = None, flip = False, debug = 0):
        '''
        subset: train, train_0.1, train_0.2, train_0.5, test
        '''
        
        split = io.load_json(split_fn)
        va_age = io.load_json(age_fn)
        va_video = io.load_json(video_fn)

        self.id_lst = split[subset]
        self.max_len = max_len
        self.age_rng = age_rng
        self.mode = mode
        self.transform = transform
        self.debug = debug
        self.flip = flip

        if self.age_rng:
            self.id_lst = [s_id for s_id in self.id_lst if self.age_rng[0] <= va_age[s_id]['age'] <= self.age_rng[1]]

        self.age_lst = [va_age[s_id] for s_id in self.id_lst]
        self.video_lst = [va_video[s_id] for s_id in self.id_lst]

        self.len = len(self.id_lst)
        print('[Video_Age_Dataset]\nsubset: %s\nsample: %d\nmode: %s\nflip: %s\nage range: %s\nsplit_fn: %s\nage_fn: %s\nvideo_fn: %s\n' %\
            (subset, self.len, self.mode, self.flip, self.age_rng, split_fn, age_fn, video_fn))


    def __len__(self):
        return 1 if self.debug == 1 else self.len

    def __getitem__(self, index):
        '''
        output data format:
            img_seq: Tensor (max_T, c, w, h)
            age: float value
            std: float value
            seq_len: int value
        '''

        if self.debug == 1:
            index = 0
        
        flip = self.flip and np.random.rand() > 0.5

        age = self.age_lst[index]['age']
        std = self.age_lst[index]['std']
        seq_len = min(self.max_len, len(self.video_lst[index]['frames']))

        img_seq = []
        for f in self.video_lst[index]['frames'][0:seq_len]:
            img = Image.open(f['image']).convert('RGB')
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            img = self.transform(img)
            img_seq.append(img)

        img_seq = torch.stack(img_seq)

        if img_seq.size(0) < self.max_len:
            pad_sz = list(img_seq.size())
            pad_sz[0] = long(self.max_len - img_seq.size(0))

            img_seq = torch.cat((img_seq, torch.zeros(pad_sz)))
        
        return img_seq, seq_len, age, std

