# preprocess YouTube Face dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


root = 'datasets/facial_video/YouTubeFaces'



def prepare_label():
    '''
    1. create clip list
    2. copy clips into /datasets/video_age/Source, and add face bounding boxes.

    clip list format:
        id          : 
        source      :
        person_id   :
        age         : -1: unlabeled
        gender      : -1: unlabeled, 1: male, 0: female
        frames      : image path of frames
        face_loc    : location of face in [x, y, width, height]
    '''

    import glob
    from collections import defaultdict
    import shutil

    dst_root = 'datasets/video_age/Source'
    dataset = 'YouTubeFace'


    clip_lst = []
    index = 0

    for idx, txt_fn in enumerate(glob.glob(os.path.join(root, 'frame_images_DB/*.labeled_faces.txt'))):

        print('creating clip list: identity %d, clip %d' % (idx, index))

        label = io.load_str_list(txt_fn, end = '\r\n')

        if label:
            # not empty file
            # format of each line: filename, [ignore], x, y, width, height, [ignore], [ignore]
            # x, y are the center of the face

            person_id = os.path.basename(txt_fn).split('.')[0]

            d = defaultdict(lambda : [])

            for line in label:

                fn, _, x, y, w, h, _, _ = line.split(',')

                fn = fn.replace('\\', '/')
                assert fn.split('/')[0] == person_id

                sub_id = fn.split('/')[1]
                loc = [int(x), int(y), int(w), int(h)]

                d[sub_id].append((fn, loc))

            for _, lst  in d.iteritems():

                clip = {
                    'id': '%s_%d' % (dataset, index),
                    'source': dataset,
                    'person_id': person_id,
                    'age': -1,
                    'gender': -1,
                    'frames': [fn for fn, _ in lst],
                    'face_loc': [loc for _, loc in lst]
                }

                clip_lst.append(clip)
                index += 1

    io.save_json(clip_lst, os.path.join(dst_root, '%s_clip.json' % dataset))


    

    num_split = 10
    num_sample_frame = 5

    io.mkdir_if_missing(os.path.join(dst_root, dataset))

    for split in range(num_split):

        split_foler = os.path.join(dst_root, dataset, 'split_%d' % split)
        io.mkdir_if_missing(split_foler)


        for idx, clip in enumerate(clip_lst):

            print('copying frames: split %d, %d / %d' % (split, idx, len(clip_lst)))

            clip_id_1, clip_id_2 = clip['id'].split('_')

            clip_folder = os.path.join(split_foler, '%s_%05d_%s' % (clip_id_1, np.random.randint(0, 100000), clip_id_2))
            io.mkdir_if_missing(clip_folder)

            for t in np.random.choice(range(len(clip['frames'])), num_sample_frame, replace = False):

                fn = clip['frames'][t]
                src_fn = os.path.join(root, 'frame_images_DB', fn)
                dst_fn = os.path.join(clip_folder, os.path.basename(fn))

                assert os.path.isfile(src_fn)

                if False:

                    shutil.copyfile(src_fn, dst_fn)
                else:

                    img = image.imread(src_fn)
                    x, y, w, h = clip['face_loc'][t]

                    w *= 2
                    h *= 3

                    img = image.draw_rectangle(img, 
                        position = [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0],
                        color = 'g')

                    image.imwrite(img, dst_fn)




if __name__ == '__main__':

    prepare_label()

