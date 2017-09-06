from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


def align_image():

    sample_lst = io.load_json('datasets/AFLW/data.json')
    
    img_root = 'datasets/AFLW/flickr'
    dst_dir = 'datasets/AFLW/Image_aligned'
    io.mkdir_if_missing(dst_dir)
    for sd in ['0', '2', '3']:
        io.mkdir_if_missing(os.path.join(dst_dir, sd))


    for idx, s in enumerate(sample_lst):

        img = image.imread(os.path.join(img_root, s['image']))
        key_points = s['key_points']

        img_a = image.align_face_19(img, key_points)

        dst_fn = os.path.join(dst_dir, s['image'])
        image.imwrite(img_a, dst_fn)

        print('%d / %d' %(idx, len(sample_lst)))


if __name__ == '__main__':

    align_image()

