# preprocess YouTube Face dataset

from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np

def image_detect_face_old():
    '''
    Detect face in each image using SenseTime SDK.

    Result will be written into a pickle file, including keypoints, pose and attributes.

    '''
    import subprocess


    # dataset = 'imdb_wiki'
    # dataset = 'morph'
    # dataset = 'megaage'
    # dataset = 'lap2016'
    # dataset = 'adience'
    dataset = 'fgnet'
    

    ## input file

    if dataset == 'imdb_wiki':
        # for IMDB-WIKI dataset
        sample_lst_fn = 'datasets/IMDB-WIKI/Annotations/imdb_wiki.json'
        img_root = '/data2/ynli/datasets/age/IMDB-WIKI/Images'
        output_fn = 'output/image_analysis/imdb_wiki_detect.json'

    elif dataset == 'morph':
        # for Morph 2 dataset
        sample_lst_fn = 'datasets/morph/Label/morph.json'
        img_root = '/data2/ynli/datasets/age/morph/sy'
        output_fn = 'output/image_analysis/morph_detect.json'

    elif dataset == 'megaface_old':
        # for MegeFace dataset

        sample_lst_fn = 'datasets/megaAge_old/Label/megaface.json'
        img_root = '/data2/ynli/datasets/age/megaAge_old/MegafaceIdentities_VGG'
        output_fn = 'output/image_analysis/megaface_old_detect.json'

    elif dataset == 'lap2016':
        # for ChaLearn_LAP_2016 dataset

        sample_lst_fn = 'datasets/LAP_2016/Label/lap.json'
        img_root = '/data2/ynli/datasets/age/LAP_2016'
        output_fn = 'output/image_analysis/lap_detect.json'

    elif dataset == 'adience':
        sample_lst_fn = 'datasets/adience/Label/adience.json'
        img_root = '/data2/ynli/datasets/age/adience/faces'
        output_fn = 'output/image_analysis/adience_detect.json'

    elif dataset == 'megaage':
        sample_lst_fn = 'datasets/megaAge/Label/megaage.json'
        img_root = '/data2/ynli/datasets/age/megaAge/Image'
        output_fn = 'output/image_analysis/megaage_detect.json'
    elif dataset == 'fgnet':
        sample_lst_fn = 'datasets/FGnet/Label/fgnet.json'
        img_root = '/data2/ynli/datasets/age/FGnet/Images'
        output_fn = 'output/image_analysis/fgnet_detect.json'

    else:
        print('invalid dataset name "%s"' % dataset)
        return



    # create intermediate image_list for SDK

    sample_lst = io.load_json(sample_lst_fn)
    img_lst = [os.path.join(img_root, s['image']) for s in sample_lst]

    for img_fn in img_lst:
        # print(img_fn)
        assert os.path.isfile(img_fn)

    img_lst_fn = os.path.join('temp', '%s_image_lst.txt' % dataset)
    rst_fn = os.path.join('temp', '%s_detect_face.txt' % dataset)
    io.save_str_list(img_lst, img_lst_fn)


    # call SenseTime SDK

    if not os.path.isfile(rst_fn):

        cwd = os.getcwd() # current working directory
        sdk_dir = 'scripts/st_SDK/st_face-7.0.0-enterprise_premium-linux-f740862/samples/c++'

        os.chdir(sdk_dir)

        command_line = [
            './detect_face.sh',
            os.path.join(cwd, img_lst_fn),
            os.path.join(cwd, rst_fn),
        ]

        print('call SenseTime SDK using:')
        print(' '.join(command_line))

        subprocess.call(command_line)

        os.chdir(cwd)


    # load face detection result
    rst_str_lst = io.load_str_list(rst_fn)
    rst_lst = []

    for idx, s in enumerate(rst_str_lst):

        print('loading detection result %d / %d' % (idx, len(rst_str_lst)))

        s = s.split(',')
        assert len(s) == 71 or len(s) == 2
        assert s[0] == os.path.join(img_root, img_lst[idx])

        if len(s) == 2:
            r = {
                'image': img_lst[idx],
                'valid': False
            }
        else:
            left, top, right, bottom = [float(v) for v in s[1:5]]
            r = {
                'image': img_lst[idx],
                'valid': True,
                'face_loc': [(left+right)/2.0, (top+bottom)/2.0, right-left+1, bottom-top+1],
                'face_pose': [float(v) for v in s[5:8]],
                'attribute': [int(v) for v in s[8:19]],
                'emotion':  [int(v) for v in s[19:29]],
                'key_points': [(float(x), float(y)) for x, y in zip(s[29::2], s[30::2])]
            }

        rst_lst.append(r)

    io.save_json(rst_lst, output_fn)


def image_analyze_face():
    '''
    analyze age dataset

    input:
        rst_lst: face detection result. created by image_detect_face()
        sample_lst: sample list of dataset
    output:
        pose_distribution
        age estimation: MAE, CA(3), CA(5)
        fail percentage
    '''



    # dataset = 'morph'
    # sample_lst = io.load_json('datasets/morph/Label/morph.json')
    # rst_lst = io.load_json('datasets/morph/Label/morph_detect_face.json')

    # dataset = 'imdb_wiki'
    # sample_lst = io.load_json('datasets/IMDB-WIKI/Annotations/imdb_wiki.json')
    # rst_lst = io.load_json('datasets/IMDB-WIKI/Annotations/imdb_wiki_detect_face.json')

    # dataset = 'megaface'
    # sample_lst = io.load_json('datasets/megaFace/Label/megaface.json')
    # rst_lst = io.load_json('datasets//megaFace/Label/megaface_detect_face.json')

    # dataset = 'lap2016'
    # sample_lst = io.load_json('datasets/LAP_2016/Label/lap.json')
    # rst_lst = io.load_json('datasets/LAP_2016/Label/lap_detect_face.json')

    dataset = 'adience'
    sample_lst = io.load_json('datasets/adience/Label/adience.json')
    rst_lst = io.load_json('datasets/adience/Label/adience_detect_face.json')


    assert len(sample_lst) == len(rst_lst)
    n_total = len(sample_lst)

    valid_idx = [i for i, r in enumerate(rst_lst) if r['valid'] == True]
    n_valid = len(valid_idx)

    sample_lst = [sample_lst[i] for i in valid_idx]
    rst_lst = [rst_lst[i] for i in valid_idx]


    # pose distribution

    bins = range(0, 91, 5)
    bins1 = range(-45, 46, 5)

    yaw_hist, _     = np.histogram([abs(r['yaw']) for r in rst_lst], bins = bins)
    pitch_hist, _   = np.histogram([r['pitch'] for r in rst_lst], bins = bins1)
    roll_hist, _    = np.histogram([abs(r['roll']) for r in rst_lst], bins = bins)

    yaw_hist = yaw_hist * 100.0  / n_valid
    pitch_hist = pitch_hist * 100.0  / n_valid
    roll_hist = roll_hist * 100.0  / n_valid

    # age estimation

    age_diff = np.array([abs(r['age'] - s['age']) for r, s in zip(rst_lst, sample_lst)])

    mae = np.mean(age_diff)
    ca3 = np.sum(age_diff <= 3.0) / len(age_diff) * 100
    ca5 = np.sum(age_diff <= 5.0) / len(age_diff) * 100


    # output

    print('dataset: %s' % dataset)
    print('\tvalid percent: %.2f%%' % (n_valid * 100.0 / n_total))
    print('\tage mae: %.2f' % mae)
    print('\tage CA3: %.2f' % ca3)
    print('\tage CA5: %.2f' % ca5)

    out_str_lst = ['%.2f,%.2f,%.2f' % t for t in zip(yaw_hist, pitch_hist, roll_hist)]
    io.save_str_list(out_str_lst, os.path.join('temp', '%s_pose_distribution.txt' % dataset))


def image_detect_face(dataset = 'megaage'):
    '''
    Detect face in image dataset

    see video_detect_face for more informaiton
    '''

    import subprocess

    if dataset == 'imdb_wiki':
        # for IMDB-WIKI dataset
        sample_lst_fn = 'datasets/IMDB-WIKI/Annotations/imdb_wiki.json'
        img_root = '/data2/ynli/datasets/age/IMDB-WIKI/Images'
        output_fn = 'output/image_analysis/imdb_wiki_detect.pkl'

    elif dataset == 'morph':
        # for Morph 2 dataset
        sample_lst_fn = 'datasets/morph/Label/morph.json'
        img_root = '/data2/ynli/datasets/age/morph/sy'
        output_fn = 'output/image_analysis/morph_detect.pkl'

    elif dataset == 'megaface_old':
        # for MegeFace dataset

        sample_lst_fn = 'datasets/megaAge_old/Label/megaface.json'
        img_root = '/data2/ynli/datasets/age/megaAge_old/MegafaceIdentities_VGG'
        output_fn = 'output/image_analysis/megaface_old_detect.pkl'

    elif dataset == 'lap2016':
        # for ChaLearn_LAP_2016 dataset

        sample_lst_fn = 'datasets/LAP_2016/Label/lap.json'
        img_root = '/data2/ynli/datasets/age/LAP_2016/Image'
        output_fn = 'output/image_analysis/lap_detect.pkl'

    elif dataset == 'adience':
        sample_lst_fn = 'datasets/adience/Label/adience.json'
        img_root = '/data2/ynli/datasets/age/adience/faces'
        output_fn = 'output/image_analysis/adience_detect.pkl'

    elif dataset == 'megaage':
        sample_lst_fn = 'datasets/megaAge/Label/megaage.json'
        img_root = '/data2/ynli/datasets/age/megaAge/Image'
        output_fn = 'output/image_analysis/megaage_detect.pkl'

    elif dataset == 'fgnet':
        sample_lst_fn = 'datasets/FGnet/Label/fgnet.json'
        img_root = '/data2/ynli/datasets/age/FGnet/Images'
        output_fn = 'output/image_analysis/fgnet_detect.pkl'

    else:
        print('invalid dataset name "%s"' % dataset)
        return


    sample_lst = io.load_json(sample_lst_fn)
    rst_dict = io.load_data(output_fn)
    for s in sample_lst:
        rst_dict[s['id']]['image'] = s['image']

    io.save_data(rst_dict, output_fn)
    return


    sample_lst = io.load_json(sample_lst_fn)
    img_lst = [os.path.join(img_root, s['image']) for s in sample_lst]

    for img_fn in img_lst:
        assert os.path.isfile(img_fn)

    img_lst_fn = os.path.join('temp', '%s_image_lst.txt' % dataset)
    io.save_str_list(img_lst, img_lst_fn)

    # call SenseTime SDK
    rst_fn = os.path.join('temp', '%s_image_detect.txt' % dataset)

    if not os.path.isfile(rst_fn):
        cwd = os.getcwd()
        sdk_dir = 'scripts/st_SDK/st_face-7.0.0-enterprise_premium-linux-f740862/samples/c++'
        os.chdir(sdk_dir)

        command_line = [
            './detect_face.sh',
            os.path.join(cwd, img_lst_fn),
            os.path.join(cwd, rst_fn),
        ]

        print('call SenseTime SDK using:')
        print(' '.join(command_line))

        subprocess.call(command_line)

        os.chdir(cwd)

    
    rst_str_lst = io.load_str_list(rst_fn)
    rst_dict = {}

    for idx, s in enumerate(rst_str_lst):
        print('loading detection result %d / %d' % (idx, len(rst_str_lst)))

        s = s.split(',')
        sample = sample_lst[idx]

        assert len(s) == 71 or len(s) == 2
        assert os.path.basename(s[0]) == os.path.basename(sample['image'])

        if len(s) == 2:
            r = {
                'image': sample['image'],
                'valid': False
            }
        else:
            left, top, right, bottom = [float(v) for v in s[1:5]]

            r = {
                'image': s[0],
                'valid': True,
                'face_loc': [(left+right)/2.0, (top+bottom)/2.0, right-left+1, bottom-top+1],
                'face_pose': [float(v) for v in s[5:8]],
                'attribute': [int(v) for v in s[8:19]],
                'emotion':  [int(v) for v in s[19:29]],
                'key_points': [(float(x), float(y)) for x, y in zip(s[29::2], s[30::2])]
            }

        rst_dict[sample['id']] = r

    io.save_data(rst_dict, output_fn)


def video_detect_face():
    '''
    Detect face in each frame of a video dataset
    
    Result includes:
        bounding box
        3-axis pose
        12 attributes
        11 emotions
        21 key-points
    '''

    import subprocess

    # dataset = 'YouTubeFace'
    # dataset = 'Celebrity-1000'
    dataset = 'COX'
    # dataset = 'IJB-B'
    # dataset = 'Celebrity-1000_pose'
    # dataset = 'Celebrity-1000_long'


    # input file

    if dataset == 'YouTubeFace':

        clip_lst = io.load_json('datasets/video_age/Source/YouTubeFace_clip.json')
        output_fn = 'output/video_analysis/YouTubeFace_detect.pkl'

        # uncropped
        # img_root = '/data2/ynli/datasets/facial_video/YouTubeFaces/frame_images_DB'

        #cropped
        img_root = '/data2/ynli/datasets/facial_video/YouTubeFaces/clips_cropped'
        

    elif dataset == 'Celebrity-1000':

        clip_lst = io.load_json('datasets/video_age/Source/Celebrity-1000_pose_clip.json')
        output_fn = 'output/video_analysis/Celebrity-1000_detect.pkl'

        # uncropped
        # img_root = '/data2/ynli/datasets/facial_video/Celebrity_1000/face_data'

        # cropped
        img_root = '/data2/ynli/datasets/facial_video/Celebrity_1000/clips_cropped'
        
    elif dataset == 'COX':

        clip_lst = io.load_json('datasets/video_age/Source/COX_sample_clip.json')
        output_fn = 'output/video_analysis/COX_detect.pkl'
        img_root = '/data2/ynli/datasets/facial_video/COX/data/video'

    else:
        print('invald dataset: %s' % dataset)
        return

    

    # create image list for SDK
    # uncropped
    img_lst = [os.path.join(img_root, frame) for c in clip_lst for frame in c['frames']]

    # cropped
    # img_lst = [os.path.join(img_root, c['id'], os.path.basename(frame)) for c in clip_lst for frame in c['frames']]


    for img_fn in img_lst:
        assert os.path.isfile(img_fn)

    img_lst_fn = os.path.join('temp', '%s_frame_lst.txt' % dataset)
    io.save_str_list(img_lst, img_lst_fn)

    # call SenseTime SDK

    rst_fn = os.path.join('temp', '%s_frame_detect.txt' % dataset)

    if not os.path.isfile(rst_fn):

        cwd = os.getcwd()
        sdk_dir = 'scripts/st_SDK/st_face-7.0.0-enterprise_premium-linux-f740862/samples/c++'
        os.chdir(sdk_dir)

        command_line = [
            './detect_face.sh',
            os.path.join(cwd, img_lst_fn),
            os.path.join(cwd, rst_fn),
        ]

        print('call SenseTime SDK using:')
        print(' '.join(command_line))

        subprocess.call(command_line)

        os.chdir(cwd)


    # load face detection result
    
    # rst_str format:
    # delimiter: ','
    # s[0]: image file
    # s[1:5]: bounding box, [left, top, right, bottom]
    # s[5:8]: pose, [yaw, pitch, roll]
    # s[8:19]: 11 attributes:
    #       s[8]: age (0-100)
    #       s[9]: gender (female 0 - 100 male)
    #       s[10]: attractive (0-100)
    #       s[11]: eyeglasses (0-100)
    #       s[12]: sunglasses (0-100)
    #       s[13]: smile (0-100)
    #       s[14]: mask (0-100)
    #       s[15]: race, 0-yellow, 1-black, 2-white
    #       s[16]: eye-open (0-100)
    #       s[17]: mouth-open (0-100)
    #       s[18]: beard (0-100)
    # s[19:29]: 10 emotions (0-100):
    #       s[19]: angry
    #       s[20]: calm
    #       s[21]: confused
    #       s[22]: disgust
    #       s[23]: happy
    #       s[24]: sad
    #       s[25]: scared
    #       s[26]: surprised
    #       s[27]: squint
    #       s[28]: scream 
    # s[29:71]: 21 key-points (x,y)

    rst_str_lst = io.load_str_list(rst_fn)
    rst_dict = {}

    index = 0

    for clip in clip_lst:
        clip_r = []

        for frame in clip['frames']:
            print('loading detection result %d / %d' % (index, len(rst_str_lst)))

            s = rst_str_lst[index].split(',')
            assert len(s) == 71 or len(s) == 2
            assert os.path.basename(s[0]) == os.path.basename(frame), '"%s" <> "%s"' % (s[0], frame)

            if len(s) == 2:

                r = {
                    'image': frame,
                    'valid': False
                }

            else:
                left, top, right, bottom = [float(v) for v in s[1:5]]

                r = {
                    'image': frame,
                    'valid': True,
                    'face_loc': [(left+right)/2.0, (top+bottom)/2.0, right-left+1, bottom-top+1],
                    'face_pose': [float(v) for v in s[5:8]],
                    'attribute': [int(v) for v in s[8:19]],
                    'emotion':  [int(v) for v in s[19:29]],
                    'key_points': [(float(x), float(y)) for x, y in zip(s[29::2], s[30::2])]
                }

            clip_r.append(r)
            index += 1

        rst_dict[clip['id']] = clip_r        

    io.save_data(rst_dict, output_fn)

def video_crop_face():
    '''
    
    crop_shift:
        x_s = int(max(0, x - w * w_scale / 2.0))
        y_s = int(max(0, y - h * h_scale / 2.0))
    '''

    # YouTubeFace dataset
    # clip_lst = io.load_json('datasets/video_age/Source/YouTubeFace_clip.json')
    # img_root = 'datasets/facial_video/YouTubeFaces/frame_images_DB'
    # output_dir = 'datasets/facial_video/YouTubeFaces/clips_cropped'

    # w_scale = 2
    # h_scale = 2

    # Celebrity-1000
    clip_lst = io.load_json('datasets/video_age/Source/Celebrity-1000_pose_clip.json')
    img_root = 'datasets/facial_video/Celebrity_1000/face_data'
    output_dir = 'datasets/facial_video/Celebrity_1000/clips_cropped'

    w_scale = 3
    h_scale = 3


    # crop frames
    io.mkdir_if_missing(output_dir)

    for idx, clip in enumerate(clip_lst):

        clip_dir = os.path.join(output_dir, clip['id'])
        io.mkdir_if_missing(clip_dir)

        for frame, face_loc in zip(clip['frames'], clip['face_loc']):

            src_fn = os.path.join(img_root, frame)
            dst_fn = os.path.join(clip_dir, os.path.basename(frame))
            assert os.path.isfile(src_fn)

            print('clip %d / %d : %s' % (idx, len(clip_lst), src_fn))

            img = image.imread(src_fn)

            x, y, w, h = face_loc

            w *= w_scale
            h *= h_scale

            img = image.crop(img, bbox = [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0])

            image.imwrite(img, dst_fn)

def test_face_alignment():

    rst_lst = io.load_json('output/image_analysis/fgnet_detect.json')

    output_dir = 'vis/face_alignment'
    io.mkdir_if_missing(output_dir)

    for idx, r in enumerate(rst_lst):
        print('%d / %d' % (idx, len(rst_lst)))
        if(r['valid']):
            img = image.imread(r['image'])
            img_out = image.align_face(img, r['key_points'])

            fn = os.path.basename(r['image'])
            fn_out = fn[0:-4] + '_aligned' + fn[-4::]
            image.imwrite(img, os.path.join(output_dir, fn))
            image.imwrite(img_out, os.path.join(output_dir, fn_out))

            
def video_analyze(dataset = 'all'):
    '''
    1. Stability of age estimation on temporal sequence
    2. Pose variation
    3. Attribute variation
    4. Correlation between age and pose/attributes
    '''

    import csv

    if dataset == 'YouTubeFace':
        sample_rate = 1
    elif dataset == 'Celebrity-1000':
        sample_rate = 3
    elif dataset == 'COX':
        sample_rate = 1
    elif dataset == 'all':
        for dataset in ['COX', 'YouTubeFace', 'Celebrity-1000']:
            video_analyze(dataset)
        return
    else:
        print('Invalid dataset: %s' % dataset)
        return

    ## load face detection result
    rst_dict = io.load_data('output/video_analysis/%s_detect.pkl' % dataset)

    ## remove the clip which has too many invalid frames (no face detected)
    min_valid_rate = 0.5
    min_valid_num = 5
    num_clip_total = len(rst_dict)

    rst_lst = [clip_r for clip_r in rst_dict.values() 
        if np.sum([f['valid'] for f in clip_r]) >= max(min_valid_rate * len(clip_r), min_valid_num)]
    num_clip_valid = len(rst_lst)

    print('[%s] total clips: %d, valid clips: %d (%.2f%%)' % (dataset, num_clip_total, num_clip_valid, 100.0 * num_clip_valid/num_clip_total))
    print('analyzing ...')

    data_mean = []
    data_var  = []
    data_cov = []
    data_cov1 = []
    data_rng = []
    for idx, clip_r in enumerate(rst_lst):
        # print('analyze clip %d / %d' % (idx, num_clip_valid))
        seq_len = len(clip_r) - 1
        clip_r = [r for r in clip_r if r['valid']]

        seq_attribute = [r['attribute'] for r in clip_r]
        seq_emotion = [r['emotion'] for r in clip_r]
        seq_yaw = [[abs(r['face_pose'][0])] for r in clip_r]

        # matrix of dimension: D * T
        # seq_age = seq_data[0, :]
        
        seq_data = np.concatenate((seq_attribute, seq_emotion, seq_yaw), axis = 1).astype(np.float).T
        seq_diff = np.abs(seq_data[:, 1::] - seq_data[:, 0:-1])

        if np.any(np.isnan(seq_data)) or np.any(np.isnan(seq_data)):
            print('NaN in data!')
            return


        # 0-order correlation
        cov = np.cov(seq_data)

        # 1-order(absolute) correlation
        cov1 = np.cov(seq_diff)

        # variation (not variance)
        var = np.sum(seq_diff, axis = 1) / seq_len / sample_rate

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


    age_bins = range(0, 101, 5)
    age_dist, _ = np.histogram(data_mean[0], bins = age_bins)
    age_dist = age_dist / np.sum(age_dist) * 100

    age_var_bins = np.concatenate((np.arange(0, 10.1, 0.25), [np.inf]))
    age_var_dist, age_var_bins = np.histogram(data_var[0], bins = age_var_bins)
    age_var_dist = age_var_dist / np.sum(age_var_dist) * 100
    # print('age var bins')
    # print(age_var_bins)

    age_rng_bins = np.concatenate((np.arange(0, 41, 5), [np.inf]))
    age_rng_dist, age_rng_bins = np.histogram(data_rng[0], bins = age_rng_bins)
    age_rng_dist = age_rng_dist / np.sum(age_rng_dist) * 100
    # print('age range bins')
    # print(age_rng_bins)

    pose_var_bins = np.concatenate((np.arange(0, 5.01, 0.1), [np.inf]))
    pose_var_dist, pose_var_bins = np.histogram(data_var[-1], bins = pose_var_bins)
    pose_var_dist = pose_var_dist / np.sum(pose_var_dist) * 100
    # print('pose var bins')
    # print(pose_var_bins)

    pose_rng_bins = np.concatenate((np.arange(0, 91, 5), [np.inf]))
    pose_rng_dist, pose_rng_bins = np.histogram(data_rng[-1], bins = pose_rng_bins)
    pose_rng_dist = pose_rng_dist / np.sum(pose_rng_dist) * 100
    # print('pose range bins')
    # print(pose_rng_bins)


    corr = ave_cov / np.sqrt(np.dot(ave_cov.diagonal().reshape(-1,1), ave_cov.diagonal().reshape(1,-1)))
    corr1 = ave_cov1 / np.sqrt(np.dot(ave_cov1.diagonal().reshape(-1,1), ave_cov1.diagonal().reshape(1,-1)))

    # output
    att_lst = [
        'Age',
        'Gender_M',
        'Attractive',
        'EyeGlass',
        'Sunglass',
        'Smile',
        'Mask',
        'Race',
        'EyeOpen',
        'MouthOpen',
        'Beard',
        'Angry',
        'Calm',
        'Confused',
        'Disgust',
        'Happy',
        'Sad',
        'Scared',
        'Surprised',
        'Squint',
        'Scream',
        'Yaw'
    ]


    output_dir = 'output/video_analysis/SDK_anylisis'
    io.mkdir_if_missing(output_dir)

    with open(os.path.join(output_dir, '%s.csv' % dataset), 'wb') as f:

        csv_writer = csv.writer(f)

        # output basic information
        csv_writer.writerow(['Dataset: %s' % dataset])
        csv_writer.writerow(['Total clips: %d'%num_clip_total, 'Valid clips: %d (%.2f%%)' % (num_clip_valid, 100.0 * num_clip_valid / num_clip_total)])
        csv_writer.writerow(['Sample rate: %d' % sample_rate])
        csv_writer.writerow([])

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

        # output pose range distribution
        csv_writer.writerow(['Pose Range', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(pose_rng_bins, pose_rng_dist)])
        csv_writer.writerow([])

        # output pose variation distribution
        csv_writer.writerow(['Pose Variation', 'Percent'])
        csv_writer.writerows([[b, d] for (b, d) in zip(pose_var_bins, pose_var_dist)])
        csv_writer.writerow([])

        # output 0-order age correlation
        csv_writer.writerow(['Attribute', 'Age Corr (0)'])
        csv_writer.writerows([[att_lst[i], c] for i, c in enumerate(corr[0])])
        csv_writer.writerow([])

        # output 1-order age correlation
        csv_writer.writerow(['Attribute', 'Age Corr (1)'])
        csv_writer.writerows([[att_lst[i], c] for i, c in enumerate(corr1[0])])
        csv_writer.writerow([])

        # output 0-order full correlation
        csv_writer.writerow([''] + att_lst)
        for i, att in enumerate(att_lst):
            csv_writer.writerow([att] + corr[i].tolist())
        csv_writer.writerow([])

        # output 1-order full correlation
        csv_writer.writerow([''] + att_lst)
        for i, att in enumerate(att_lst):
            csv_writer.writerow([att] + corr1[i].tolist())
        csv_writer.writerow([])


def image_align_face(dataset = 'megaage', key_point_num = 21):
    '''
    use face detect result to align face and crop image to size (178, 218)
    '''

    assert key_point_num in {3, 21}

    if dataset == 'imdb_wiki':
        # for IMDB-WIKI dataset
        img_root = '/data2/ynli/datasets/age/IMDB-WIKI/Images'
        output_dir = '/data2/ynli/datasets/age/IMDB-WIKI/Images_aligned_%d' % key_point_num
        rst_dict_fn = 'output/image_analysis/imdb_wiki_detect.pkl'

    elif dataset == 'lap2016':
        # for ChaLearn_LAP_2016 dataset
        img_root = '/data2/ynli/datasets/age/LAP_2016/Image'
        output_dir = '/data2/ynli/datasets/age/LAP_2016/Image_aligned_%d' % key_point_num
        rst_dict_fn = 'output/image_analysis/lap_detect.pkl'

    elif dataset == 'megaage':
        img_root = '/data2/ynli/datasets/age/megaAge/Image'
        output_dir = '/data2/ynli/datasets/age/megaAge/Image_aligned_%d' % key_point_num
        rst_dict_fn = 'output/image_analysis/megaage_detect.pkl'

    else:
        print('invalid dataset name "%s"' % dataset)
        return



    rst_dict = io.load_data(rst_dict_fn)

    # build directory structure
    tree = os.walk(img_root)
    for t in tree:
        src_dir = t[0]
        dst_dir = src_dir.replace(img_root, output_dir, 1)
        print(dst_dir)
        io.mkdir_if_missing(dst_dir)

    # align and crop image
    for idx, rst in enumerate(rst_dict.values()):
        print('aligning image [%s, %d points]: %d / %d' % (dataset, key_point_num, idx, len(rst_dict)))
        src_fn = os.path.join(img_root, rst['image'])
        dst_fn = os.path.join(output_dir, rst['image'])

        img = image.imread(src_fn)
        if rst['valid']:
            if key_point_num == 3:
                img = image.align_face_3(img, rst['key_points'])
            else:
                img = image.align_face_21(img, rst['key_points'])
        else:
            img = image.resize(img, (178, 218))

        image.imwrite(img, dst_fn)





if __name__ == '__main__':

    ## image
    # image_detect_face(sys.argv[1])
    image_align_face(sys.argv[1])

    ## video()
    # video_detect_face()
    # video_crop_face()

    # video_analyze()


    ## test
    # test_face_alignment()