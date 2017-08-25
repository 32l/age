# preprocess IJB-B dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


root = 'datasets/facial_video/IJB-B'


def prepare_label():
    '''
    1. create clip list, each clip only has on frame
    2. copy clips into /datasets/vidoe_age/Source
    '''

    import csv
    import shutil

    source_name = 'IJBB'
    dst_root = 'datasets/video_age/Source'

    clip_lst = []
    index = 0

    # load person name

    subject_dict = {}

    with open(os.path.join(root, 'protocol', 'ijbb_subject_names.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')

        # ignore the first line
        _ = reader.next()

        for line in reader:
            subject_dict[line[1]] = line[0]

    print('load %d subject names' % len(subject_dict))

    # load clips

    clip_lst = []
    index = 0

    with open(os.path.join(root, 'protocol', 'ijbb_1N_probe_video.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')

        # ignore the first line
        _ = reader.next()

        for line in reader:

            # subject = line[1]
            # frame = line[2]
            # video = line[3]
            # face_loc = [int(x) for x in line[5:9]] # left, top, width, height
            # begin_frame = line[4]

            # print('createing clip list: %d' % index)

            loc = [float(x) for x in line[5:9]] # left, top, width, height
            loc[0] += loc[2] / 2
            loc[1] += loc[3] / 2

            clip = {
                'id': '%s_%d' % (source_name, index),
                'source': source_name,
                'person_id': subject_dict[line[1]],
                'age': -1,
                'gender': -1,
                'frames': [line[2]],
                'face_loc': [loc], # [x, y, width, height]
                'video': line[3],
                'begin_frame': int(line[-1]) # the frame index in video
            }

            clip_lst.append(clip)
            index += 1

    print('load %d clips' % len(clip_lst))
    io.save_json(clip_lst, os.path.join(dst_root, '%s_clip.json' % source_name))
    


    #  create data for labeling
    num_split = 10

    io.mkdir_if_missing(os.path.join(dst_root, source_name))

    for split in range(num_split):

        split_folder = os.path.join(dst_root, source_name, 'split_%d' % split)
        io.mkdir_if_missing(split_folder)

        for idx, clip in enumerate(clip_lst):

            print('copying frames: split %d, %d / %d' % (split, idx, len(clip_lst)))

            clip_id_1, clip_id_2 = clip['id'].split('_')

            clip_folder = os.path.join(split_folder, '%s_%05d_%s' % (clip_id_1, np.random.randint(0, 100000), clip_id_2))
            io.mkdir_if_missing(clip_folder)

            # each clip only has one frame now
            fn = clip['frames'][0]
            src_fn = os.path.join(root, fn)
            dst_fn = os.path.join(clip_folder, os.path.basename(fn))

            assert os.path.isfile(src_fn)

            if False:
                shutil.copyfile(src_fn, dst_fn)
            else:
                img = image.imread(src_fn)
                x, y, w, h = clip['face_loc'][0]

                w *= 1.5
                h *= 1.5

                img = image.draw_rectangle(img,
                    position = [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0],
                    color = 'g')

                image.imwrite(img, dst_fn)


if __name__ == '__main__':

    prepare_label()
