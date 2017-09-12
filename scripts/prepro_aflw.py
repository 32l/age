from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np

def create_label():

    sample_lst = io.load_json('datasets/AFLW/data.json')

    for i in xrange(len(sample_lst)):
        for j in range(3):
            sample_lst[i]['pose'][j] = sample_lst[i]['pose'][j]

    io.mkdir_if_missing('datasets/AFLW/Label')
    io.save_json(sample_lst, 'datasets/AFLW/Label/aflw.json')

    train_lst = [s for s in sample_lst if s['id'].startswith('train')]
    test_lst = [s for s in sample_lst if s['id'].startswith('test')]

    io.save_json(train_lst, 'datasets/AFLW/Label/aflw_train.json')
    io.save_json(test_lst, 'datasets/AFLW/Label/aflw_test.json')


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


def dist():

    import csv

    sample_lst = io.load_json('datasets/AFLW/data.json')

    bins = np.arange(-90,91, 10.0)
    yaw_lst = [s['pose'][0] / np.pi * 180 for s in sample_lst]
    h_yaw, _ = np.histogram(yaw_lst, bins = bins)
    h_yaw = h_yaw / np.sum(h_yaw) * 100.0

    pitch_lst = [s['pose'][1] / np.pi * 180 for s in sample_lst]
    h_pitch, _ = np.histogram(pitch_lst, bins = bins)
    h_pitch = h_pitch / np.sum(h_pitch) * 100.0


    fn = 'output/attribute_analysis/aflw_pose_dist.csv'
    with open(fn, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows([['%d'%b, '%.2f'%y, '%.2f'%p] for (b, y, p) in zip(bins, h_yaw, h_pitch)])


        


if __name__ == '__main__':

    # align_image()
    # create_label()
    dist()

