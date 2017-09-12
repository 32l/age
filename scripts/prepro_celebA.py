from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


def align_image():
    '''
    Aligned images are provided in dataset
    '''
    pass

def create_label():

    root = 'datasets/CelebA'
    img_root = os.path.join(root, 'Image_aligned')

    data = io.load_str_list(os.path.join(root, 'Anno', 'list_attr_celeba.txt'), end = '\r\n')
    partition = io.load_str_list(os.path.join(root, 'Eval', 'list_eval_partition.txt'), end = '\r\n')

    num_sample = int(data[0])
    attr_name_lst = data[1].split()

    sample_lst = []
    
    for idx, (s_str, p_str) in enumerate(zip(data[2::], partition)):
        print(idx)
        s_str = s_str.split()
        p_str = p_str.split()

        assert s_str[0] == p_str[0] # image name

        attr = [1 if l == '1' else 0 for l in s_str[1::]]

        p = 'trainval' if p_str[1] == '0' or p_str[1] == '1' else 'test'

        sample_lst.append({
            'id': '%s_%d' % (p, idx),
            'image': s_str[0],
            'attr': attr
            })

    assert len(sample_lst) == num_sample

    io.mkdir_if_missing(os.path.join(root, 'Label'))
    io.save_str_list(attr_name_lst, os.path.join(root, 'Label', 'attr_name_lst.txt'))
    io.save_json(sample_lst, os.path.join(root, 'Label', 'celeba.json'))
    io.save_json([s for s in sample_lst if s['id'].startswith('trainval')], os.path.join(root, 'Label', 'celeba_train.json'))
    io.save_json([s for s in sample_lst if s['id'].startswith('test')], os.path.join(root, 'Label', 'celeba_test.json'))




if __name__ == '__main__':

    # align_image()
    create_label()

