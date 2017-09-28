# preprocess Video Age dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


def create_age_label():
    # configs
    version = '1.0'
    num_split = 6
    min_anno = 5
    subset_lst = [
        {'dataset': 'YouTubeFace', 'clip_fn': 'datasets/video_age/Source/YouTubeFace_clip.json'},
        {'dataset': 'Celebrity-1000', 'clip_fn': 'datasets/video_age/Source/Celebrity-1000_pose_clip.json'}
    ]

    # sample dict
    all_samples = {}

    # samples[id] = {'age', 'std', 'anno'}
    

    for subset in subset_lst:
        samples = {}
        dataset = subset['dataset']
        clips = io.load_json(subset['clip_fn'])
        clips = {c['id']:c for c in clips}

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


    output_fn = os.path.join('datasets/video_age/Labels/v%s_age.json' % version)
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

def alignment():

    va_video = io.load_json('datasets/video_age/Labels/v1.0_video.json')
    out_root = 'datasets/video_age/Videos'

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
    io.save_json(va_video, 'datasets/video_age/Labels/v1.0_video.json')

def create_identity_label():

    version = '1.0'
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

    

    io.save_json(all_samples, 'datasets/video_age/Labels/v%s_person.json'%version)

def create_split():

    version = '1.0'
    train_perc = 0.7

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

    version = '1.0'
    
    va_age = io.load_json('datasets/video_age/Labels/v%s_age.json' % version)
    va_video = io.load_json('datasets/video_age/Labels/v%s_video.json' % version)
    va_person = io.load_json('datasets/video_age/Labels/v%s_person.json' % version)
    va_split = io.load_json('datasets/video_age/Labels/v%s_split.json' % version)

    for subset in ['train', 'test']:
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


def video_analyze():

    import csv

    # load experiment results
    print('loading experiment results ...')
    # rst_age = io.load_data('models/age_va_4.0.3/video_test_rst.pkl')
    # rst_age = io.load_data('models/age_pre_2.2/video_test_rst.pkl')
    rst_age = io.load_data('models/age_morph_3.2/video_test_rst.pkl')

    rst_pose = io.load_data('models/pose_4.0.2n/video_test_rst.pkl')
    rst_attr = io.load_data('models/attr_1.0.3/video_test_rst.pkl')

    id_lst = rst_age.keys()
    print('sample number: %d' % len(id_lst))


    # load attribute list

    attr_lst = io.load_str_list('datasets/CelebA/Label/attr_name_lst.txt')
    assert len(attr_lst) == len(rst_attr.values()[0][0])

    pose_lst = ['yaw', 'pitch']
    assert len(pose_lst) == len(rst_pose.values()[0][0])

    item_lst = ['age'] + pose_lst + attr_lst + ['random']

    # analyze
    print('analyzing ...')

    data_mean = []
    data_var  = []
    data_cov = []
    data_cov1 = []
    data_rng = []

    for idx, s_id in enumerate(id_lst):

        seq_age = rst_age[s_id] # list [seq_len]
        seq_pose = rst_pose[s_id] # list [seq_len, pose_dim]
        seq_attr = rst_attr[s_id] # list [seq_len, num_attr]


        seq_len = len(seq_age)

        seq_age = np.array(seq_age).reshape(seq_len, 1)
        seq_pose = np.array(seq_pose) / np.pi * 180.
        seq_random = np.random.rand(seq_len, 1)


        seq_data = np.concatenate((seq_age, seq_pose, seq_attr, seq_random), axis = 1).astype(np.float).T
        seq_diff = np.abs(seq_data[:, 1::] - seq_data[:, 0:-1])

        seq_data[1, :] = np.abs(seq_data[1, :]) # get absolute value of yaw

        # 0-order correlation
        cov = np.cov(seq_data)

        # 1-order(absolute) correlation
        cov1 = np.cov(seq_diff)

        # variation (not variance)
        var = np.sum(seq_diff, axis = 1) / seq_len

        # average value
        mean = np.mean(seq_data, axis = 1)

        # range
        rng = np.max(seq_data, axis = 1) - np.min(seq_data, axis = 1)
        
        data_mean.append(mean)
        data_var.append(var)
        data_cov.append(cov)
        data_cov1.append(cov1)
        data_rng.append(rng)

    data_mean = np.array(data_mean, np.float).T
    data_var = np.array(data_var, np.float).T
    data_rng = np.array(data_rng, np.float).T
    ave_cov = np.mean(data_cov, axis = 0)
    ave_cov1 = np.mean(data_cov1, axis =0)

    age_bins = range(0, 71, 5)
    age_dist, _ = np.histogram(data_mean[0], bins = age_bins)
    age_dist = age_dist / np.sum(age_dist) * 100

    age_var_bins = np.concatenate((np.arange(0, 10.1, 0.25), [np.inf]))
    age_var_dist, age_var_bins = np.histogram(data_var[0], bins = age_var_bins)
    age_var_dist = age_var_dist / np.sum(age_var_dist) * 100
    # print('age var bins')
    # print(age_var_bins)

    age_rng_bins = np.concatenate((np.arange(0, 21, 2), [np.inf]))
    age_rng_dist, age_rng_bins = np.histogram(data_rng[0], bins = age_rng_bins)
    age_rng_dist = age_rng_dist / np.sum(age_rng_dist) * 100
    # print('age range bins')
    # print(age_rng_bins)

    # yaw_var_bins = np.concatenate((np.arange(0, 5.01, 0.1), [np.inf]))
    yaw_var_bins = np.concatenate((np.arange(0, 15.01, 0.3), [np.inf]))
    yaw_var_dist, yaw_var_bins = np.histogram(data_var[1], bins = yaw_var_bins)
    # yaw_var_dist, yaw_var_bins = np.histogram(data_var[1])
    yaw_var_dist = yaw_var_dist / np.sum(yaw_var_dist) * 100
    # print('yaw var bins')
    # print(yaw_var_bins)
    # print(yaw_var_dist)

    yaw_rng_bins = np.concatenate((np.arange(0, 91, 5), [np.inf]))
    yaw_rng_dist, yaw_rng_bins = np.histogram(data_rng[1], bins = yaw_rng_bins)
    yaw_rng_dist = yaw_rng_dist / np.sum(yaw_rng_dist) * 100
    # print('yaw range bins')
    # print(yaw_rng_bins)



    corr = ave_cov / np.sqrt(np.dot(ave_cov.diagonal().reshape(-1,1), ave_cov.diagonal().reshape(1,-1)))
    corr1 = ave_cov1 / np.sqrt(np.dot(ave_cov1.diagonal().reshape(-1,1), ave_cov1.diagonal().reshape(1,-1)))

    # output
    output_dir = 'output/video_age_analysis'
    io.mkdir_if_missing(output_dir)

    output_fn = os.path.join(output_dir, 'video_age_1.0_morph_3.2.csv')

    with open(output_fn, 'wb') as f:
        csv_writer = csv.writer(f)

        # output age distribution
        csv_writer.writerow(['Age', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(age_bins, age_dist)])
        csv_writer.writerow([])

        # output age range distribution
        csv_writer.writerow(['Age Range', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(age_rng_bins, age_rng_dist)])
        csv_writer.writerow([])
        
        # outut age variation distribution
        csv_writer.writerow(['Age Variation', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(age_var_bins, age_var_dist)])
        csv_writer.writerow([])

        # output yaw range distribution
        csv_writer.writerow(['Yaw Range', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(yaw_rng_bins, yaw_rng_dist)])
        csv_writer.writerow([])

        # output yaw variation distribution
        csv_writer.writerow(['Yaw Variation', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(yaw_var_bins, yaw_var_dist)])
        csv_writer.writerow([])

        # output 0-order age correlation
        csv_writer.writerow(['Attribute', 'Age Corr (0)'])
        csv_writer.writerows([[item_lst[i], c] for i, c in enumerate(corr[0])])
        csv_writer.writerow([])

        # output 1-order age correlation
        csv_writer.writerow(['Attribute', 'Age Corr (1)'])
        csv_writer.writerows([[item_lst[i], c] for i, c in enumerate(corr1[0])])
        csv_writer.writerow([])

        # output 0-order full correlation
        csv_writer.writerow([''] + item_lst)
        for i, att in enumerate(item_lst):
            csv_writer.writerow([att] + corr[i].tolist())
        csv_writer.writerow([])

        # output 1-order full correlation
        csv_writer.writerow([''] + item_lst)
        for i, att in enumerate(item_lst):
            csv_writer.writerow([att] + corr1[i].tolist())
        csv_writer.writerow([])



if __name__ == '__main__':

    # create_age_label()
    # create_video_label()
    # alignment()
    # create_identity_label()
    # create_split()
    # create_image_dataset()


    video_analyze()
    # pass
    
