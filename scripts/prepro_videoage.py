# preprocess Video Age dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


def create_age_label():
    # configs
    version = '2.0'
    num_split = 6
    min_anno = 5
    subset_lst = [
        {'dataset': 'YouTubeFace', 'clip_fn': 'datasets/video_age/Source/YouTubeFace_clip.json'},
        {'dataset': 'Celebrity-1000', 'clip_fn': 'datasets/video_age/Source/Celebrity-1000_pose_clip_1.json'}
    ]

    # sample dict
    all_samples = {}

    # samples[id] = {'age', 'std', 'anno'}
    

    for subset in subset_lst:
        samples = {}
        dataset = subset['dataset']
        clips = io.load_json(subset['clip_fn'])
        
        # build frame_dict
        if dataset == 'Celebrity-1000':
            frame_dict = {(c['id'] + '_' + os.path.basename(f)): idx for c in clips for f,idx in zip(c['frames'], c['face_idx'])}

        for split in range(num_split):
            split_str = 'split_%d' % split
            anno_dir = 'datasets/video_age/Annotations/%s/%s' % (dataset, split_str)
            print('%s: split_%d ...' % (dataset, split))

            for anno_fn in os.listdir(anno_dir):
                
                anno = io.load_json(os.path.join(anno_dir, anno_fn))
                try:
                    age = float(anno['objects']['face2'][0]['attributes'][u'\u5e74\u9f84']['value'])

                except:
                    # annotation missing
                    continue

                # split_k/[dataset]_[random_code]_[id_code]
                s1, s = anno['image']['rawFilePath'].split('/')
                s2, _, s3 = s.split('_')

                assert s1 == split_str, '%s != %s' % (s1, split_str)
                assert s2 == dataset, '%s != %s' % (s2, dataset)

                s_id = s2 + '_' + s3


                # add 2017.10.1 ################
                if dataset == 'Celebrity-1000':
                    f_id = s_id + '_' + anno['image']['rawFilename']
                    assert f_id in frame_dict
                
                    if frame_dict[f_id] != 0:
                        continue


                #################################


                if s_id not in samples:
                    samples[s_id] = {
                        'id': s_id,
                        'age': -1.,
                        'std': -1.,
                        'anno': [], 
                        'raw': [[] for _ in range(num_split)]
                    }

                samples[s_id]['raw'][split].append(age)

        for s_id, samp in samples.iteritems():

            samp['anno'] = [np.median(l).tolist() for l in samp['raw'] if l]
            samp['age'] = np.mean(samp['anno']).round().tolist()
            samp['std'] = np.std(samp['anno']).tolist()

            samples[s_id] = samp

        num_sample = len(samples)
        samples = {s_id: samp for s_id, samp in samples.iteritems() if len(samp['anno']) >= min_anno}
        num_sample_valid = len(samples)

        print('%s: total %d, valid: %d' % (dataset, num_sample, num_sample_valid))

        all_samples.update(samples)


    output_fn = os.path.join('datasets/video_age/Labels/v%s_age_pre.json' % version)
    io.save_json(all_samples, output_fn)

def create_video_label():

    # configs
    version = '1.0'
    subset_lst = [
        {
            'dataset': 'YouTubeFace',
            'clip_fn': 'datasets/video_age/Source/YouTubeFace_clip.json',# created by prepare_label() in prepro_ytf.py
            'detect_fn': 'output/video_analysis/YouTubeFace_detect.pkl', # created by video_detect_face() in prepro_general.py
            'img_root': 'datasets/facial_video/YouTubeFaces/frame_images_DB',
            'max_len': 17,
            'sample_rate': 3,
            'crop_scale': 2 # see video_crop_face() in prepro_general. This scale value is used to restore key-point coordinates in the original image 
        },

        {
            'dataset': 'Celebrity-1000', 
            'clip_fn': 'datasets/video_age/Source/Celebrity-1000_pose_clip.json',
            'detect_fn': 'output/video_analysis/Celebrity-1000_detect.pkl',
            'img_root': 'datasets/facial_video/Celebrity_1000/face_data',
            'max_len': 17,
            'sample_rate': 1,
            'crop_scale': 3,
        }
    ]

    num_pose = 21

    
    def _interp_pose(pose_seq, t_seq, t_seq_cont):
        assert len(pose_seq) == len(t_seq)

        pose_trace = []
        for i in xrange(num_pose):
            x_seq = [p[i][0] for p in pose_seq]
            x_seq_cont = np.interp(t_seq_cont, t_seq, x_seq)

            y_seq = [p[i][1] for p in pose_seq]
            y_seq_cont = np.interp(t_seq_cont, t_seq, y_seq)
            pose_trace.append(zip(x_seq_cont, y_seq_cont))

        pose_seq_cont = zip(*pose_trace)
        return pose_seq_cont


    def _restore_key_point(pose, face_loc, crop_scale):

        x, y, w, h = face_loc

        # left-top of cropped image
        left = int(max(0, x - w * crop_scale / 2.))
        top = int(max(0, y - h * crop_scale / 2.))

        for i, (xp, yp) in enumerate(pose):
            pose[i] = (xp + left, yp + top)

        return pose

    all_samples = {}

    np.random.seed(0)
    for subset in subset_lst:

        samples = {}
        dataset = subset['dataset']
        rst_dict = io.load_data(subset['detect_fn'])
        clips = io.load_json(subset['clip_fn'])
        clips = {c['id']: c for c in clips}

        sr = subset['sample_rate']
        ml = (subset['max_len'] - 1) * sr + 1
        cs = subset['crop_scale']

        for s_id, clip in clips.iteritems():
            print('%s: %s' % (dataset, s_id))
            rst = rst_dict[s_id]
            # key-point interpolation

            t_seq = [i for i in range(len(clip['frames'])) if rst[i]['valid']]
            t_seq_cont = range(len(clip['frames']))
            pose_seq = [_restore_key_point(rst[i]['key_points'], clip['face_loc'][i], cs)
                            for i in range(len(clip['frames'])) if rst[i]['valid']]

            if len(pose_seq) > 0:
                pose_seq_cont = _interp_pose(pose_seq, t_seq, t_seq_cont)
            else:
                pose_seq_cont = [None] * len(t_seq_cont)
            

            # crop short clips
            s = np.random.randint(max(1, len(clip['frames']) - ml + 1)) # random sample
            e = min(s+ml, len(clip['frames']))

            samples[s_id] = {
                'id': s_id,
                'frames': [{
                    'org_image': os.path.join(subset['img_root'], clip['frames'][i]),
                    'face_loc': clip['face_loc'][i],
                    'key_points': pose_seq_cont[i],
                    'mean_pose': False} for i in range(s, e, sr)]
            }

            
        # compute mean pose
        # mean_pose_scale = [x_scal, y_scal]
        # x = x_c + w * x_scal
        
        print('computing mean pose ...')
        pose_scale = []
        for samp in samples.values():
            for f in samp['frames']:
                if f['key_points'] is not None:
                    x, y, w, h = f['face_loc']
                    scale = [[(p[0]-x)/w, (p[1]-y)/h] for p in f['key_points']]
                    pose_scale.append(scale)

        pose_scale = np.array(pose_scale)
        mean_pose_scale = np.mean(pose_scale, axis = 0)
        print(mean_pose_scale)

        for s_id, samp in samples.iteritems():
            for t, f in enumerate(samp['frames']):
                if f['key_points'] is None:
                    x, y, w, h = f['face_loc']
                    samples[s_id]['frames'][t]['key_points'] = [[x+w*mean_pose_scale[i][0], y+h*mean_pose_scale[i][1]] for i in range(num_pose)]
                    samples[s_id]['frames'][t]['mean_pose'] = True
        

        all_samples.update(samples)

    output_fn = os.path.join('datasets/video_age/Labels/v%s_video.json' % version)
    io.save_json(all_samples, output_fn)

def create_video_label_1():
    # configs
    version = '2.0'
    subset_lst = [
        {
            'dataset': 'YouTubeFace',
            'clip_fn': 'datasets/video_age/Source/YouTubeFace_clip.json',# created by prepare_label() in prepro_ytf.py
            'detect_fn': 'output/video_analysis/YouTubeFace_detect_1.pkl', # created by video_detect_face() in prepro_general.py
            'img_root': 'datasets/facial_video/YouTubeFaces/frame_images_DB',
            'sample_rate': 3,
            'crop_scale': [1.5, 2] # see video_crop_face() in prepro_general. This scale value is used to restore key-point coordinates in the original image 
        },

        {
            'dataset': 'Celebrity-1000', 
            'clip_fn': 'datasets/video_age/Source/Celebrity-1000_pose_clip_1.json',
            'detect_fn': 'output/video_analysis/Celebrity-1000_detect_1.pkl',
            'img_root': 'datasets/facial_video/Celebrity_1000/face_data',
            'sample_rate': 1,
            'crop_scale': [3, 3],
        }
    ]

    min_len = 8 # min clip length (after sampling)
    max_len = 17 # max clip length (after sampling)
    max_num = 3  # max clip number sampled from the same video
    max_gap = 1  # max length of gap (contineous bad frames)

   
    def _interp_pose(pose_seq, t_seq, t_seq_cont):

        num_pose = 21
        assert len(pose_seq) == len(t_seq)
        pose_trace = []
        for i in xrange(num_pose):
            x_seq = [p[i][0] for p in pose_seq]
            x_seq_cont = np.interp(t_seq_cont, t_seq, x_seq)

            y_seq = [p[i][1] for p in pose_seq]
            y_seq_cont = np.interp(t_seq_cont, t_seq, y_seq)
            pose_trace.append(zip(x_seq_cont, y_seq_cont))

        pose_seq_cont = zip(*pose_trace)
        return pose_seq_cont

    def _restore_key_point(pose, face_loc, crop_scale):

        x, y, w, h = face_loc

        # left-top of cropped image
        left = int(max(0, x - w * crop_scale[0] / 2.))
        top = int(max(0, y - h * crop_scale[1] / 2.))

        for i, (xp, yp) in enumerate(pose):
            pose[i] = (xp + left, yp + top)

        return pose

    all_samples = {}
    all_samples_sdk_rst = {}
    log = ''

    np.random.seed(0)
    for subset in subset_lst:
        samples = subset['dataset']
        samples = {}
        dataset = subset['dataset']
        rst_dict = io.load_data(subset['detect_fn'])
        clips = io.load_json(subset['clip_fn'])
        clips = {c['id']: c for c in clips}
        sr = subset['sample_rate']
        max_l = (max_len - 1) * sr + 1
        min_l = (min_len - 1) * sr + 1
        cs = subset['crop_scale']

        for s_id, clip in clips.iteritems():
            print('%s: %s' % (dataset, s_id))
            rst = rst_dict[s_id]

            seq_len = len(clip['frames'])

            # key-point interpolation
            t_seq = [i for i in range(seq_len) if rst[i]['valid']]
            t_seq_cont = range(seq_len)
            pose_seq = [_restore_key_point(rst[i]['key_points'], clip['face_loc'][i], cs)
                            for i in range(seq_len) if rst[i]['valid']]
            if len(pose_seq) > 0:
                pose_seq_cont = _interp_pose(pose_seq, t_seq, t_seq_cont)
            else:
                pose_seq_cont = [None] * len(t_seq_cont)

            interp_seq = [False if rst[i]['valid'] else True for i in range(seq_len)]


            # create age_seq as a reference of video clip sampling
            age_seq = np.array([r['attribute'][0] if r['valid'] else -1 for r in rst])


            # video segmentation
            # valid_seq[i] == 1 if i-th frame is valid, else 0
            valid_seq = np.ones(seq_len)
            invalid_counter = 0
            for i in xrange(seq_len):
                if rst[i]['valid']:
                    invalid_counter = 0
                else:
                    invalid_counter += 1
                    if invalid_counter > max_gap:
                        valid_seq[(i - max_gap):(i+1)] = 0


            # sample video clip
            for n in range(max_num):

                # l_seq[i] is the length of the clip candidates starting at i-th frame
                l_seq = np.zeros(seq_len, dtype = np.int)
                # score_seq[i] is the score of clip candidates starting at i-th frame.
                # the score is defined as age_rng + 0.01 * clip_len
                score_seq = - np.ones(seq_len)
                
                for i in xrange(seq_len - min_l + 1):
                    next_invalid = np.where(valid_seq[i:(i+max_l)] == 0)[0]
                    l = next_invalid[0] if len(next_invalid) > 0 else min(max_l, seq_len - i)
                    if l >= min_l:
                        l_seq[i] = l
                        score_seq[i] = age_seq[i:(i+l)].max() - age_seq[i:(i+l)].min() + 0.01 * l

                # select clip candidates with largest score
                i_s = np.argmax(score_seq)

                if score_seq[i_s] < 0:
                    break
                else:
                    i_e = i_s + l_seq[i_s]
                    assert (min_l <= i_e - i_s <= max_l) and (i_e <= seq_len)

                    new_id = s_id + '_' + str(n)

                    samples[new_id] = {
                        'id': new_id,
                        'org_id': s_id,
                        'frames': [{
                            'org_image': os.path.join(subset['img_root'], clip['frames'][i]),
                            'face_loc': clip['face_loc'][i],
                            'key_points': pose_seq_cont[i],
                            'pose_interp': interp_seq[i]
                        } for i in range(i_s, i_e, sr)]
                    }

                    all_samples_sdk_rst[new_id] = {
                        'age': [r['attribute'][0] if r['valid'] else None for r in rst[i_s:i_e:sr]],
                        'pose': [r['face_pose'][0:2] if r['valid'] else None for r in rst[i_s:i_e:sr]],
                        'attr': [r['attribute'][1::] + r['emotion'] if r['valid'] else None for r in rst[i_s:i_e:sr]],
                        'valid': [r['valid'] for r in rst[i_s:i_e:sr]]
                    }

                # update valid_seq
                valid_seq[i_s:i_e] = 0


        all_samples.update(samples)
        log += 'dataset: %s, sampled %d clips from %d videos\n' % (dataset, len(samples), len(clips))

    print(log)
    output_fn = os.path.join('datasets/video_age/Labels/v%s_video.json' % version)
    io.save_json(all_samples, output_fn)


    rst_output_fn = 'output/video_analysis/video_age_v%s_detect.pkl' % version
    io.save_data(all_samples_sdk_rst, rst_output_fn)


    # update age label
    va_age_pre = io.load_json('datasets/video_age/Labels/v%s_age_pre.json' % version)
    va_age = {}
    for s_id, samp in all_samples.iteritems():
        org_id = samp['org_id']
        if org_id in va_age_pre:
            samp_age = va_age_pre[org_id].copy()
            samp_age['id'] = s_id
            va_age[s_id] = samp_age

    io.save_json(va_age, 'datasets/video_age/Labels/v%s_age.json' % version)

    # update person label
    va_person_pre = io.load_json('datasets/video_age/Labels/v%s_person_pre.json' % version)
    va_person = {}
    for s_id, samp in all_samples.iteritems():
        org_id = samp['org_id']
        samp_person = va_person_pre[org_id].copy()
        samp_person['id'] = s_id
        va_person[s_id] = samp_person

    io.save_json(va_person, 'datasets/video_age/Labels/v%s_person_pre.json' % version)


def interp_sdk_rst():

    sdk_rst = io.load_data('output/video_analysis/video_age_v2.0_detect.pkl.copy')
    for s_id, rst in sdk_rst.iteritems():
        if False in rst['valid']:
            seq_len = len(rst['valid'])
            t_seq = [t for t in range(seq_len) if rst['valid'][t]]
            t_seq_cont = range(seq_len)

            # age
            age_seq = [rst['age'][t] for t in t_seq]
            rst['age'] = np.interp(t_seq_cont, t_seq, age_seq).tolist()

            # pose
            pose_seq_cont = np.zeros((2, seq_len))
            for i in range(2):
                pose_seq = [rst['pose'][t][i] for t in t_seq]
                pose_seq_cont[i] = np.interp(t_seq_cont, t_seq, pose_seq)
            rst['pose'] = pose_seq_cont.T.tolist()

            # attribute
            attr_seq_cont = np.zeros((20, seq_len))
            for i in range(20):
                attr_seq = [rst['attr'][t][i] for t in t_seq]
                attr_seq_cont[i] = np.interp(t_seq_cont, t_seq, attr_seq)
            rst['attr'] = attr_seq_cont.T.tolist()

            sdk_rst[s_id] = rst

    io.save_data(sdk_rst, 'output/video_analysis/video_age_v2.0_detect.pkl')

def alignment():

    va_video = io.load_json('datasets/video_age/Labels/v2.0_video.json')
    out_root = 'datasets/video_age/Videos_v2.0'
    io.mkdir_if_missing(out_root)

    for s_id, samp in va_video.iteritems():
        print('%s' % s_id)
        out_dir = os.path.join(out_root, s_id)
        io.mkdir_if_missing(out_dir)

        for t, f in enumerate(samp['frames']):

            src_fn = f['org_image']
            dst_fn = os.path.join(out_dir, '%d.jpg' % t)

            print(src_fn)
            print(dst_fn)
            print('')

            img = image.imread(src_fn)
            img = image.align_face_21(img, f['key_points'])

            image.imwrite(img, dst_fn)

            va_video[s_id]['frames'][t]['image'] = dst_fn
    io.save_json(va_video, 'datasets/video_age/Labels/v2.0_video.json')

def create_identity_label():

    version = '2.0'
    subset_lst = [
        {'dataset': 'YouTubeFace', 'clip_fn': 'datasets/video_age/Source/YouTubeFace_clip.json'},
        {'dataset': 'Celebrity-1000', 'clip_fn': 'datasets/video_age/Source/Celebrity-1000_pose_clip.json'}
    ]


    all_samples = {}
    name_to_id = {}

    for subset in subset_lst:
        dataset = subset['dataset']
        clips = io.load_json(subset['clip_fn'])
        samples = {}

        for clip in clips:
            s_id = clip['id']
            name = clip['person_id']

            name = name.replace('_', ' ').lower().strip()

            if name not in name_to_id:
                name_to_id[name] = '%d' % len(name_to_id)

            samples[s_id] = {
                'id': s_id,
                'person_id': name_to_id[name],
                'person_name': name
            }

        print('%d' % len(name_to_id))
        all_samples.update(samples)

    io.save_json(all_samples, 'datasets/video_age/Labels/v%s_person_pre.json'%version)

def create_split():

    version = '2.0'
    train_perc = 0.5

    train_split = {
        'train_0.1': 0.1 ,
        'train_0.2': 0.2 ,
        'train_0.5': 0.5 , 
    }

    va_age = io.load_json('datasets/video_age/Labels/v%s_age.json' % version)
    va_person = io.load_json('datasets/video_age/Labels/v%s_person.json' % version)
    va_video = io.load_json('datasets/video_age/Labels/v%s_video.json' % version)


    # build person-age table
    pa_table = {}
    for s_id, samp in va_age.iteritems():
        p_id = va_person[s_id]['person_id']
        if p_id not in pa_table:
            pa_table[p_id] = []
        pa_table[p_id].append(samp['age'])

    pa_table = {p_id: np.mean(v) for p_id, v in pa_table.iteritems()}

    age_bins = range(0,101,10)

    # split
    split = {
        'train': [],
        'test': []
    }
    for k in train_split.keys():
        split[k] = []

    np.random.seed(0)
    for age_l, age_h in zip(age_bins, age_bins[1::]):
        persons = [p_id for p_id, age in pa_table.iteritems() if age_l <= age < age_h]

        if len(persons) == 0:
            continue

        np.random.shuffle(persons)

        num_person = len(persons)
        
        persons_train = set(persons[0:int(num_person * train_perc)])
        persons_test = set(persons[int(num_person * train_perc)::])

        sample_train = [s_id for s_id, _ in va_age.iteritems() if va_person[s_id]['person_id'] in persons_train]
        sample_test = [s_id for s_id, _ in va_age.iteritems() if va_person[s_id]['person_id'] in persons_test]

        split['train'] += sample_train
        split['test'] += sample_test

        np.random.shuffle(sample_train)
        for k, p in train_split.iteritems():
            split[k] += sample_train[0:int(len(sample_train) * p)]

        
        print('age from %d to %d: total sample %d (train %d, test %d); total person %d (train %d, test %d)' %\
            (age_l, age_h, len(sample_train) + len(sample_test), len(sample_train), len(sample_test),
            num_person, len(persons_train), len(persons_test)))

    
    # cleaning
    for subset, id_lst in split.iteritems():
        new_id_lst = []
        
        for s_id in id_lst:
            if len(va_video[s_id]['frames']) > 1:
                new_id_lst.append(s_id)

        split[subset] = new_id_lst

    split['all'] = split['train'] + split['test']

    io.save_json(split, 'datasets/video_age/Labels/v%s_split.json' % version)

def create_image_dataset():
    '''
    create files to build image dataset (modules.datasets.Image_Age_Dataset)
    frames are regarded as individual images.
    '''

    version = '2.0'
    
    va_age = io.load_json('datasets/video_age/Labels/v%s_age.json' % version)
    va_video = io.load_json('datasets/video_age/Labels/v%s_video.json' % version)
    va_person = io.load_json('datasets/video_age/Labels/v%s_person.json' % version)
    va_split = io.load_json('datasets/video_age/Labels/v%s_split.json' % version)

    for subset in ['train', 'test', 'train_0.1', 'train_0.2', 'train_0.5']:
        sample_lst = []

        for s_id in va_split[subset]:
            samp = va_video[s_id]

            for t, f in enumerate(samp['frames']):
                sample_lst.append({
                    'id': s_id + '_f' + str(t),
                    'person_id': va_person[s_id]['person_id'],
                    'age': va_age[s_id]['age'],
                    'std': va_age[s_id]['std'],
                    'image': f['image'],
                    })

        print('%s: %d' % (subset, len(sample_lst)))
        io.save_json(sample_lst, 'datasets/video_age/Labels/v%s_image_%s.json' % (version, subset))




if __name__ == '__main__':

    # create_age_label()
    # create_identity_label()
    # create_video_label()
    # create_video_label_1()
    # alignment()
    # interp_sdk_rst()
    # create_split()
    # create_image_dataset()

    # pass
    
