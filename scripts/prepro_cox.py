# preprocess COX dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


root = 'datasets/facial_video/COX'

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

    from collections import defaultdict
    import shutil

    dst_root = 'datasets/video_age/Source'
    dataset = 'COX'
    person_id_lst = io.load_str_list(os.path.join(root, 'test_file', 'cox_id_list.txt'))


    clip_lst = []
    index = 0

    for idx, person_id in enumerate(person_id_lst):

    	print('creating clip for subject %d / %d' % (idx, len(person_id_lst)))

    	for cam_id in [1, 2]:

    		video_dir = os.path.join(root, 'data', 'video', 'cam%d' % cam_id, person_id)

    		frame_lst = os.listdir(video_dir)
    		frame_lst.sort()

    		# only use the tail of the sequence, where the area of person is large
    		frame_lst = frame_lst[-60::]
    		frame_lst = [os.path.join('cam%d' % cam_id, person_id, fn) for fn in frame_lst]

    		clip = {
    			'id': '%s_%d' % (dataset, index),
    			'source': dataset,
    			'person_id': person_id,
    			'age': -1,
    			'gender': -1,
    			'frames': frame_lst,
    			'face_loc': [None] * len(frame_lst),
    			'cam_id': cam_id
    		}

    		clip_lst.append(clip)
    		index += 1

    io.save_json(clip_lst, os.path.join(dst_root, '%s_clip.json' % dataset))


    # sample one video for each subject
    clip_dict = defaultdict(lambda : [])

    for clip in clip_lst:
    	clip_dict[clip['person_id']].append(clip)

    clip_samp_lst = []

    for k, v in clip_dict.iteritems():
    	np.random.shuffle(v)
    	clip_samp_lst.append(v[0])

    io.save_json(clip_samp_lst, os.path.join(dst_root, '%s_sample_clip.json' % dataset))



    # output

    num_split = 10
    num_sample_frame = 5

    io.mkdir_if_missing(os.path.join(dst_root, dataset))

    for split in range(num_split):

    	split_dir = os.path.join(dst_root, dataset, 'split_%d' % split)
    	io.mkdir_if_missing(split_dir)

    	for idx, clip in enumerate(clip_samp_lst):

    		print('copying frames: split %d, %d / %d' % (split, idx, len(clip_samp_lst)))

    		clip_id_1, clip_id_2 = clip['id'].split('_')
    		clip_dir = os.path.join(split_dir, '%s_%05d_%s' % (clip_id_1, np.random.randint(0, 100000), clip_id_2))
    		io.mkdir_if_missing(clip_dir)

    		

    		for t in np.random.choice(range(len(clip['frames'])), num_sample_frame, replace = False):
    			
    			fn = clip['frames'][t]
    			src_fn = os.path.join(root, 'data', 'video', fn)
    			dst_fn = os.path.join(clip_dir, os.path.basename(fn))

    			assert os.path.isfile(src_fn)

    			shutil.copyfile(src_fn, dst_fn)



if __name__ == '__main__':

	prepare_label()

