from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np

def create_label_megaface():

    root = 'datasets/megaFace'
    img_root = os.path.join(root, 'MegafaceIdentities_VGG')

    img_map = io.load_json(os.path.join(root, 'name_map.json'))

    train_str_lst = io.load_str_list(os.path.join(root, 'train_caffe_com100.txt'))
    test_str_lst = io.load_str_list(os.path.join(root, 'test_caffe_com100.txt'))

    train_lst = []
    for s in train_str_lst:
        s = s.split(' ')
        assert len(s) == 73

        sample = {
            'image': img_map[s[0]],
            'age': float(s[1]),
            'gender': -1,
            'person_id': -1,
            'distribution': [float(v) for v in s[3::]]
        }

        assert os.path.isfile(os.path.join(img_root, sample['image']))

        train_lst.append(sample)


    test_lst = []
    for s in test_str_lst:
        s = s.split(' ')
        assert len(s) == 73

        sample = {
            'image': img_map[s[0]],
            'age': float(s[1]),
            'gender': -1,
            'person_id': -1,
            'distribution': [float(v) for v in s[3::]]
        }

        assert os.path.isfile(os.path.join(img_root, sample['image']))

        test_lst.append(sample)

    sample_lst = train_lst + test_lst

    io.save_json(sample_lst, os.path.join(root, 'Label', 'megaface.json'))
    io.save_json(train_lst, os.path.join(root, 'Label', 'megaface_train.json'))
    io.save_json(test_lst, os.path.join(root, 'Label', 'megaface_test.json'))

def create_label_lap2016():
    import csv

    root = 'datasets/LAP_2016'


    train_lst = []
    with open(os.path.join(root, 'Label', 'train_gt.csv'), 'rb') as csvfile:

        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')

        # ignore the first line
        _ = reader.next()

        for line in reader:

            sample = {
                'image': 'train/' + line[0],
                'age': float(line[1]),
                'std': float(line[2]),
                'person_id': -1,
            }
            train_lst.append(sample)

    val_lst = []
    with open(os.path.join(root, 'Label', 'valid_gt.csv'), 'rb') as csvfile:

        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')

        # ignore the first line
        _ = reader.next()

        for line in reader:

            sample = {
                'image': 'valid/' + line[0],
                'age': float(line[1]),
                'std': float(line[2]),
                'person_id': -1,
            }
            val_lst.append(sample)

    test_lst = []
    with open(os.path.join(root, 'Label', 'test_gt.csv'), 'rb') as csvfile:

        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')

        # ignore the first line
        _ = reader.next()

        for line in reader:

            sample = {
                'image': 'test/' + line[0],
                'age': float(line[1]),
                'std': float(line[2]),
                'person_id': -1,
            }
            test_lst.append(sample)

    lap_lst = train_lst + val_lst + test_lst

    print('train: %d' % len(train_lst))
    print('val: %d' % len(val_lst))
    print('test: %d' % len(test_lst))

    io.save_json(lap_lst, os.path.join(root, 'Label', 'lap.json'))
    io.save_json(train_lst, os.path.join(root, 'Label', 'lap_train.json'))
    io.save_json(val_lst, os.path.join(root, 'Label', 'lap_val.json'))
    io.save_json(test_lst, os.path.join(root, 'Label', 'lap_test.json'))

def create_label_adience():

    root = 'datasets/adience'
    sample_lst = []

    for n in xrange(5):
        # data are splited into 5 folds

        str_lst = io.load_str_list(os.path.join(root, 'cross_validation', 'fold_%d_data.txt' % n))
        for s in str_lst:
            s = s.split('\t')

            image = '%s/coarse_tilt_aligned_face.%s.%s' % (s[0], s[2], s[1])
            assert os.path.isfile(os.path.join(root, 'faces', image))

            if s[4] == 'm':
                gender = 1
            elif s[4] == 'f':
                gender = 0
            else:
                gender = -1

            left, top, width, height = [float(c) for c in s[5:9]]
            x = left + width / 2.0
            y = top + height / 2.0


            sample_lst.append({
                'image': image,
                'age': -1,
                'age_str': s[3],
                'gender': gender,
                'face_loc': [x, y, width, height],
                'person_id': s[2]
                })

    print('load %d image' % len(sample_lst))

    io.save_json(sample_lst, os.path.join(root, 'Label', 'adience.json'))

def create_label_megaage():

    root = 'datasets/megaAge'

    sub_lst = {}

    io.mkdir_if_missing(os.path.join(root, 'Label'))

    for subset in ['train', 'test']:

        fn_lst = io.load_str_list(os.path.join(root, 'List', 'BMVC70_name_%s.txt' % subset))
        age_lst = io.load_str_list(os.path.join(root, 'List', 'BMVC70_mean_%s.txt' % subset))
        dis_lst = io.load_str_list(os.path.join(root, 'List', 'BMVC70_dis_%s.txt' % subset), end = ' \n')
        assert len(fn_lst) == len(age_lst) == len(dis_lst)

        sample_lst = []

        for fn, age, dis in zip(fn_lst, age_lst, dis_lst):

            sample_lst.append({
                    'image': fn,
                    'age': float(age),
                    'gender': -1,
                    'person_id': -1,
                    'dis': [float(v) for v in dis.split(' ')]
                })

        sub_lst[subset] = sample_lst
        io.save_json(sample_lst, os.path.join(root, 'Label', 'megaage_%s.json' % subset))


    sample_lst = sub_lst['train'] + sub_lst['test']
    io.save_json(sample_lst, os.path.join(root, 'Label', 'megaage.json'))


def create_label_fgnet():

    root = 'datasets/FGnet'

    anno = io.load_str_list(os.path.join(root, 'fgnet_id_age.txt'))[1::]
    sample_lst = []

    for s in anno:
        s = s.split()
        sample_lst.append({
            'image': s[0],
            'age': float(s[2]),
            'gender': int(s[3]), # 0-female,1-male
            'person_id': s[1]
            })

    print('FGnet load %d samples' % len(sample_lst))
    io.mkdir_if_missing(os.path.join(root, 'Label'))
    io.save_json(sample_lst, os.path.join(root, 'Label', 'fgnet.json'))

    # age_label = io.load_str_list(os.path.join(root, 'age_label.txt'), end = '\r\n')
    # age_list = io.load_str_list(os.path.join(root, 'age_list.txt'), end = '\r\n')

    # sample_lst = []
    # for label, name in zip(age_label, age_list):
    #     sample_lst.append({
    #         'image': name,
    #         'age': float(label),
    #         'gender': -1,
    #         'person_id': -1
    #         })
    # io.save_json(sample_lst, os.path.join(root, 'Label', 'fgnet_1.json'))




if __name__ == '__main__':


    # create_label_megaface()
    # create_label_lap2016()
    # create_label_adience()
    # create_label_megaage()
    create_label_fgnet()