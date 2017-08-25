# preprocess Celebrity-1000 dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


root = 'datasets/facial_video/Celebrity_1000'


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
    dataset = 'Celebrity-1000'

    
    # load celebrity names
    celeb_name_lst = io.load_str_list(os.path.join(root, 'names.txt'))
    celeb_name_lst = [s.split(' ', 1)[1] for s in celeb_name_lst]


    # helper functions

    def _load_celebrity1000_annotation(fn):
        '''
        load the annotation of the first face from data file
        '''
        anno = io.load_str_list(fn, end = '\r\n')

        # face number

        num = int(anno[1])
        if num == 0:
            return None, None, -1

        # load face score
        score = int(anno[7])

        # load bbox
        line = anno[11] # [x y width height]
        
        x, y, width, height = [float(s) for s in line[1:-1].split(' ')]

        x += width / 2.0
        y += height / 2.0

        # load face pose
        pitch = float(anno[15])
        yaw   = float(anno[17])
        roll  = float(anno[19])


        face_loc = [x, y, width, height]
        face_pose = [yaw, pitch, roll]

        return face_loc, face_pose, score



    # load all clips
    clip_lst = []
    index = 0

    full_lst_fn = os.path.join(dst_root, '%s_clip.json' % dataset)
    if os.path.isfile(full_lst_fn):
    # if False:
        clip_lst = io.load_json(full_lst_fn)
    else:
        for celeb_idx, celeb_name in enumerate(celeb_name_lst):
            celeb_idx  += 1

            celeb_dir = os.path.join(root, 'face_data', '%04d' % celeb_idx)

            for video_id in os.listdir(celeb_dir):
                for clip_id in os.listdir(os.path.join(celeb_dir, video_id)):

                    print('%d_%s_%s' % (celeb_idx, video_id, clip_id))

                    clip = {
                        'id': '%s_%d' % (dataset, index),
                        'source': dataset,
                        'person_id': unicode(celeb_idx),
                        'video_id': video_id,
                        'clip_id': clip_id,
                        'age': -1,
                        'gender': -1,
                        'frames': [],
                        'face_loc': [],
                        'face_pose': [],
                        'face_score': []
                    }


                    # load frames
                    clip_dir = os.path.join(celeb_dir, video_id, clip_id)
                    frame_lst = glob.glob('%s/*_ori.jpg' % clip_dir)
                    t_lst = [int(os.path.basename(f)[0:-8]) for f in frame_lst]
                    t_lst.sort()

                    for t in t_lst:
                        
                        fn = os.path.join('%04d'%celeb_idx, video_id, clip_id, '%d_ori.jpg' % t)
                        anno_fn = os.path.join(root, 'face_data', '%04d'%celeb_idx, video_id, clip_id, '%d.jpg.txt' % t)
                        assert os.path.isfile(os.path.join(root, 'face_data', fn))


                        face_loc, face_pose, face_score = _load_celebrity1000_annotation(anno_fn)

                        clip['frames'].append(fn)
                        clip['face_loc'].append(face_loc)
                        clip['face_pose'].append(face_pose)
                        clip['face_score'].append(face_score)

                    if len(clip['frames']) > 0:
                        clip_lst.append(clip)
                        index += 1

        io.save_json(clip_lst, full_lst_fn)
    print('load %d clips' % len(clip_lst))


    clip_dict = defaultdict(lambda : [])
    for clip in clip_lst:
        key = clip['person_id'] + '_' + clip['video_id']
        clip_dict[key].append(clip)


    # select sequence with large face
    clip_long_lst = []
    min_face_area = 5000

    for k, c_lst in clip_dict.iteritems():
        
        # remove non-face clips
        c_lst = [c for c in c_lst if np.min(c['face_score']) > 0]

        # remove clips with small face
        c_lst_v = [c for c in c_lst if np.mean([l[2] * l[3] for l in c['face_loc']]) >= min_face_area]
        
        if len(c_lst_v) == 0:
            c_lst_v = c_lst

        c_lst_v.sort(key = lambda c: len(c['frames']), reverse = True)
        clip_long_lst.append(c_lst_v[0])

    print('clip_long_lst: %d clips, average length: %f' % (len(clip_long_lst), np.mean([len(c['frames']) for c in clip_long_lst])))
    io.save_json(clip_long_lst, os.path.join(dst_root, '%s_long_clip.json' % dataset))

    # select sequence with large pose change.

    clip_pose_lst = []
    min_length = 12

    for k, c_lst in clip_dict.iteritems():

        # remove non-face clips
        c_lst = [c for c in c_lst if np.min(c['face_score']) > 0]

        # remove clips that are too short or with small face
        c_lst_v = [c for c in c_lst if len(c['frames']) > min_length 
            and np.mean([l[2] * l[3] for l in c['face_loc']]) >= min_face_area]
        
        if len(c_lst_v) == 0:
            c_lst_v = c_lst

        c_lst_v.sort(key = lambda c: np.var([y for y,_,_ in c['face_pose'] if y != None]), reverse = True)
        clip_pose_lst.append(c_lst_v[0])

    print('clip_pose_lst: %d clips, average length: %f' % (len(clip_pose_lst), np.mean([len(c['frames']) for c in clip_pose_lst])))
    io.save_json(clip_pose_lst, os.path.join(dst_root, '%s_pose_clip.json' % dataset))



    # output
    num_split = 10
    num_sample_frame = 5

    for clip_lst, appendix in [(clip_long_lst, 'long'), (clip_pose_lst, 'pose')]:
        
        io.mkdir_if_missing(os.path.join(dst_root, dataset + '_' + appendix))

        for split in range(num_split):

            split_folder = os.path.join(dst_root, dataset + '_' + appendix, 'split_%d' % split)
            io.mkdir_if_missing(split_folder)

            for idx, clip in enumerate(clip_lst):

                print('copying frames: split %d, %d / %d' % (split, idx, len(clip_lst)))

                clip_id_1, clip_id_2 = clip['id'].split('_')

                clip_folder = os.path.join(split_folder, '%s_%05d_%s' % (clip_id_1, np.random.randint(0, 100000), clip_id_2))
                io.mkdir_if_missing(clip_folder)

                t_lst = np.random.choice(range(len(clip['frames'])), min(num_sample_frame, len(clip['frames'])), replace = False)
                for t in t_lst:

                    fn = clip['frames'][t]
                    src_fn = os.path.join(root, 'face_data', fn)
                    dst_fn = os.path.join(clip_folder, os.path.basename(fn))

                    assert os.path.isfile(src_fn)

                    if False:

                        shutil.copyfile(src_fn, dst_fn)
                    else:

                        img = image.imread(src_fn)
                        x, y, w, h = clip['face_loc'][t]

                        w *= 3
                        h *= 3

                        # img = image.draw_rectangle(img, 
                        #     position = [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0],
                        #     color = 'g')

                        img = image.crop(img, bbox = [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0])

                        image.imwrite(img, dst_fn)



if __name__ == '__main__':

    prepare_label()




    