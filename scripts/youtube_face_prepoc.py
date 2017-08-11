from __future__ import print_function, division

import sys
import os
import util.io as io
import numpy as np


def organize_label_morph():

    img_root = 'datasets/morph/sy'
    old_img_root = 'image/morph/sy'

    for subset in {'train', 'test'}:

        img_lst = io.load_str_list('datasets/morph/label/%s_imglist'%subset)
        code = io.load_str_list('datasets/morph/label/%s_code'%subset)[1::]

        assert len(img_lst) == len(code)

        img_lst = [img.split()[0].replace(old_img_root, img_root) for img in img_lst]
        gen_lst = [int(c.split()[0]) for c in code]
        age_lst = [int(c.split()[1]) for c in code]

        person_id_lst = [img.split('/')[-1][0:6] for img in img_lst]

        data = [{'image': img, 'gender': g, 'age': a, 'person_id': p} 
            for img, g, a, p in zip(img_lst, gen_lst, age_lst, person_id_lst)]

        data_lst = ['%s %s %d %d' % (img, p, g, a) 
            for (img, p, g, a) in zip(img_lst, person_id_lst, gen_lst, age_lst)]


        io.save_json(data, 'datasets/morph/Label/%s.json'%subset)
        io.save_str_list(data_lst, 'datasets/morph/Label/%s.lst'%subset)

        for d in data:
            if not os.path.isfile(d['image']):
                print(d['image'])

def organize_label():

    dst_lst = io.load_str_list('datasets/youtube_face/Meta/dst.lst')
    old_root = 'image'
    root = 'datasets/youtube_face/Images'

    for dst in dst_lst:
        print(dst)

        lst = io.load_str_list('datasets/youtube_face/Label/%s.lst' % dst)
        lst = [l.replace(old_root, root) for l in lst]

        img_lst = []

        for l in lst:
            img, gen, age = l.split()
            gen, age = int(gen), int(age)

            img_lst.append({'image': img, 'gender': gen, 'age': age})

        io.save_json(img_lst, 'datasets/youtube_face/Label/%s.json' % dst)
        io.save_str_list(lst, 'datasets/youtube_face/Label/%s.lst' % dst)

def organize_clip_label():
    from collections import defaultdict
    import re

    dst_lst = io.load_str_list('datasets/youtube_face/Meta/dst.lst')

    old_root = 'image'
    root = 'datasets/youtube_face/Images'

    max_frame_interval = 5


    for dst in dst_lst:
        print(dst)

        img_lst = io.load_json('datasets/youtube_face/Label/%s.json' % dst)
        data_dict = defaultdict(lambda : [])

        for idx, img in enumerate(img_lst):
            
            s = img['image'].split('/')

            ts = re.split('\.|-', s[-1])[0]
            # print(s)
            # print(ts)

            img['time_stamp'] = int(ts)

            # the image path is like "a/b/c.jpg"
            # now we consider that sequence frames must be under same "b" folder
            # (maybe they could be under same "a")

            key = s[-2]

            data_dict[key].append(img)

        clip_lst = []

        def _new_clip():
            clip =  {
                'id': '',
                'frames': [],
                'gender': -1,
                'age': -1,
                'suspicious': 0,
            }
            return clip

        for k, v in data_dict.iteritems():
            # v is a list of image

            v.sort(key = lambda x: x['time_stamp'])
            
            t = v[0]['time_stamp']
            n = 0

            clip = _new_clip()

            for i, img in enumerate(v):

                if img['time_stamp'] - t > max_frame_interval:
                    clip['dataset'] = dst
                    clip['id']      = '%s_%s_%d' % (dst, k, n)
                    clip['gender']  = clip['frames'][0]['gender']
                    clip['age']     = clip['frames'][0]['age']
                    clip['suspicious'] = 1

                    clip_lst.append(clip)
                    clip = _new_clip()

                    n += 1

                clip['frames'].append(img)
                t = img['time_stamp']

            if len(clip['frames']) > 0:
                clip['dataset'] = dst
                clip['id']      = '%s_%s_%d' % (dst, k, n)
                clip['gender']  = clip['frames'][0]['gender']
                clip['age']     = clip['frames'][0]['age']
                clip['suspicious'] = 0 if n == 0 else 1
                clip_lst.append(clip)

        io.save_json(clip_lst, 'datasets/youtube_face/Label/%s_clip.json'%dst)

        num_clip = len(clip_lst)
        num_susp = np.sum([c['suspicious'] for c in clip_lst])
        num_long = np.sum([len(c['frames']) > 10 for c in clip_lst])

        print('\tnum_clip: %d' % num_clip)
        print('\tnum_susp: %d (%f%%)' % (num_susp, 100.0*num_susp/num_clip))
        print('\tnum_long: %d (%f%%)' % (num_long, 100.0*num_long/num_clip))

    # combine all dataset
    clip_lst = []
    for dst in dst_lst:
        clip_lst.extend(io.load_json('datasets/youtube_face/Label/%s_clip.json' % dst))

    io.save_json(clip_lst, 'datasets/youtube_face/Label/youtube_face_clip.json')


def age_hist_morph():

    bins = range(0, 101, 5)

    img_lst = []
    for subset in ['train', 'test']:
        img_lst.extend(io.load_json('datasets/morph/Label/%s.json' % subset))

    print('load %d images' % len(img_lst))

    age_m = [img['age'] for img in img_lst if img['gender'] == 0]
    age_f = [img['age'] for img in img_lst if img['gender'] == 1]

    age_hist_m, _ = np.histogram(age_m, bins = bins)
    age_hist_f, _ = np.histogram(age_f, bins = bins)

    output = ['%d-%d\t%d\t%d' % (bins[i], bins[i+1], hm, hf) 
        for i, (hm, hf) in enumerate(zip(age_hist_m, age_hist_f))]

    io.save_str_list(output, 'output/age_hist_image_morph.txt')

def age_hist(name = 'new_big_pose'):

    bins = range(0, 101, 5)

    # load data

    dst_lst = io.load_str_list('datasets/youtube_face/Meta/dst.lst')
    if name == 'all':
        pass
    else:
        dst_lst = [dst for dst in dst_lst if dst.startswith(name)]




    clip_lst = []

    for dst in dst_lst:
        print('laoding %s ...' % dst)
        clip_lst.extend(io.load_json('datasets/youtube_face/Label/%s_clip.json' % dst))

    print('load %d clips' % len(clip_lst))

    
    age_c_m = [c['age'] for c in clip_lst if c['gender'] == 0]
    age_c_f = [c['age'] for c in clip_lst if c['gender'] == 1]
    weight_c_m = [len(c['frames']) for c in clip_lst if c['gender'] == 0]
    weight_c_f = [len(c['frames']) for c in clip_lst if c['gender'] == 1]

    # clip_level
    age_hist_c_m,_ = np.histogram(age_c_m, bins = bins)
    age_hist_c_f,_ = np.histogram(age_c_f, bins = bins)

    output = ['%d-%d\t%d\t%d' % (bins[i], bins[i+1], hm, hf) 
        for i, (hm, hf) in enumerate(zip(age_hist_c_m, age_hist_c_f))]

    io.save_str_list(output, 'output/age_hist_clip_%s.txt'%name)

    
    # image_level
    age_hist_i_m,_ = np.histogram(age_c_m, bins = bins, weights = weight_c_m)
    age_hist_i_f,_ = np.histogram(age_c_f, bins = bins, weights = weight_c_f)

    output = ['%d-%d\t%d\t%d' % (bins[i], bins[i+1], hm, hf) 
        for i, (hm, hf) in enumerate(zip(age_hist_i_m, age_hist_i_f))]

    io.save_str_list(output, 'output/age_hist_image_%s.txt'%name)

def clip_length_distribution_raw(name = 'new_big_pose'):

    bins = [0, 5, 10, 20, 50, 100, 200, np.inf]

    # load data
    dst_lst = io.load_str_list('datasets/youtube_face/Meta/dst.lst')

    if name == 'all':
        pass
    else:
        dst_lst = [dst for dst in dst_lst if dst.startswith(name)]

    clip_lst = []
    for dst in dst_lst:
        print('laoding %s ...' % dst)
        clip_lst.extend(io.load_json('datasets/youtube_face/Label/%s_clip.json' % dst))

    print('load %d clips' % len(clip_lst))


    # clip length hist

    len_lst = [len(c['frames']) for c in clip_lst]
    print('load %d clips (> 10 frames)' % np.sum(np.array(len_lst) > 10))

    len_hist, _ = np.histogram(len_lst, bins = bins)
    len_i_hist, _  = np.histogram(len_lst, bins = bins, weights = len_lst)

    bins_str = [str(b) if b != np.inf else '' for b in bins]
    output = ['%s-%s\t%d\t%d' % (bins_str[i], bins_str[i+1], h, h_img) 
    for i, (h, h_img) in enumerate(zip(len_hist, len_i_hist))]

    io.save_str_list(output, 'output/clipL_hist_%s.txt'%name)


def detect_artifact():

    from shutil import copyfile

    # load data
    clip_lst = io.load_json('datasets/youtube_face/Label/youtube_face_clip.json')

    score_lst = io.load_str_list('output/artifact_score.txt') # created by scripts/detect_artifact.m
    score_lst = [tuple(l.split()) for l in score_lst]

    score_dict = {k:(float(s) if s != 'nan' else -1.0) for k, s in score_lst}

    # output dir
    output_dir = 'output/artifact'
    io.mkdir_if_missing(output_dir)

    # check artifact score

    threshold = 0.2

    n_neg_clip = 0
    n_neg_image = 0
    n_image = 0

    for c_idx, clip in enumerate(clip_lst):

        print('[%.2f%%]' % (c_idx *100.0 / len(clip_lst)))

        for i_idx, img in enumerate(clip['frames']):
            n_image += 1
            s = score_dict[img['image']]
            clip_lst[c_idx]['frames'][i_idx]['artifact_score'] = s

            if s < threshold and s > 0:
                n_neg_image += 1

                fn = os.path.join(output_dir, os.path.basename(img['image']))
                copyfile(img['image'], fn)


        ss = np.array([img['artifact_score'] for img in clip['frames']])

        if np.all(ss < 0):
            s_c = -1.0
        else:
            s_c = np.min(ss)

        clip_lst[c_idx]['artifact_score'] = s_c

        if s < threshold and s > 0:
            n_neg_clip += 1

    io.save_json(clip_lst, 'datasets/youtube_face/Label/youtube_face_clip.json')

    print('negative images: %d / %d (%f%%)' % (n_neg_image, n_image, n_neg_image * 100.0/n_image))
    print('negative clips: %d / %d (%f%%)' % (n_neg_clip, len(clip_lst), n_neg_clip * 100.0/len(clip_lst)))


def serialize_clip_lst(name = 'youtube_face'):

    # load clip list

    # this is for identity grouping using SenseTime SDK

    if name == 'youtube_face':
        clip_lst = io.load_json('datasets/youtube_face/Label/youtube_face_clip.json')
        print('load youtube_face')
    else:
        dst_lst = io.load_str_list('datasets/youtube_face/Meta/dst.lst')
        clip_lst = []
        for dst in dst_lst:
            if dst.startswith(name):
                clip_lst.extend(io.load_json('datasets/youtube_face/Label/%s_clip.json' % dst))
                print('load %s' % dst)

    # re-organize the clips in to: dataset->group->clips
    print('organizing clips')
    
    from collections import defaultdict

    clip_dict = defaultdict(lambda : defaultdict(lambda: []))

    for clip in clip_lst:

        dst = clip['dataset']
        # the id format is [dst]_[group_idx]_[clip_idx_in_group]
        group = clip['id'].split('_')[-2]

        clip_dict[dst][group].append(clip)

    clip_dict = {dst:
                    [{'group': int(g), 'clips': cs} for g, cs in group_dict.iteritems()] 
                for dst, group_dict in clip_dict.iteritems()}

    for dst in clip_dict.keys():
        clip_dict[dst].sort(key = lambda x:x['group'])


    # output clip list into txt file
    print('output txt file')

    output_fn = 'output/clip.dat'
    output_lst = []

    output_lst.append(str(len(clip_dict))) # the number of datasets

    for dst, group_lst in clip_dict.iteritems():
        output_lst.append(dst)
        output_lst.append(str(len(group_lst)))

        for g in group_lst:
            cs = g['clips']
            output_lst.append(str(len(cs)))

            for clip in cs:
                output_lst.append(clip['id'])
                output_lst.append(str(len(clip['frames'])))

                for img in clip['frames']:
                    output_lst.append(img['image'])

    io.save_str_list(output_lst, output_fn)



    # output_fn = 'output/clip.lst'
    # output_lst = []

    # output_lst.append(str(len(clip_lst)))
    # dst2idx = {}
    
    # for clip in clip_lst:
    #     if clip['dataset'] not in dst2idx:
    #         dst2idx[clip['dataset']] = str(len(dst2idx))

    #     output_lst.append(dst2idx[clip['dataset']])
    #     output_lst.append(clip['id'])
    #     output_lst.append(str(len(clip['frames'])))

    #     for img in clip['frames']:
    #         output_lst.append(img['image'])

    # io.save_str_list(output_lst, output_fn)

def add_identity():

    clip_lst = io.load_json('datasets/youtube_face/Label/youtube_face_clip.json')
    id_str_lst = io.load_str_list('output/face_identity_0.5.txt')

    identity_dict = {}

    for s in id_str_lst:
        clip_id, valid, face_id = s.split('\t')

        # valid: 0-invalid, 1-valid
        # face_id: -1(no face in image), -2(multiple identities), other(id)
        identity_dict[clip_id] = face_id


    for c_idx, clip in enumerate(clip_lst):
        clip_id = clip['id']
        
        try:
            face_id = identity_dict[clip_id]

        except:

            face_id = '-3'
            print('no identity info for clip: (%d) %s' % (c_idx, clip_id))

        clip_lst[c_idx]['identity'] = face_id


    io.save_json(clip_lst, 'datasets/youtube_face/Label/youtube_face_clip_id0.5.json')

def image_number_per_identity():
    # draw a bar chart: image number - identity

    from collections import defaultdict

    invalid_id = {'-1', '-2'}

    clip_lst = io.load_json('datasets/youtube_face/Label/youtube_face_clip_id0.7.json')

    id2num = defaultdict(int)

    n_clip = 0

    for c in clip_lst:
        if _validate_clip(c, t_artifact = 0.2, t_clip_length = 10):
            id2num[c['identity']] += len(c['frames'])
            n_clip += 1



    num_id_lst = [(k, n) for k, n in id2num.iteritems()]

    num_id_lst.sort(key = lambda x:x[1], reverse = True)

    output = ['%s\t%d'%(identity, num) for identity, num in num_id_lst]

    print(n_clip)
    print(len(num_id_lst))

    # io.save_str_list(output, 'output/imageNum_per_identity.txt')


def clip_length_distribution():

    clip_lst = io.load_json('datasets/youtube_face/Label/youtube_face_clip_id0.7.json')

    clip_lst = [c for c in clip_lst if _validate_clip(c, t_artifact = 0.2, t_clip_length = 10)]
    clip_length_lst = [len(c['frames']) for c in clip_lst]

    bins = [0, 10, 20, 50, 100, 200, 500, 1000, np.inf]
    hist, _ = np.histogram(clip_length_lst, bins = bins)

    for i, h in enumerate(hist):
        print('%3s - %3s: %d' % (str(bins[i]), str(bins[i+1]), h))

    print(max(clip_length_lst))




def _validate_clip(clip, t_artifact = 0.2, t_clip_length = 10):

    if clip['identity'] in {'-1', '-2'}:
        return False

    if clip['artifact_score'] <= t_artifact:
        return False

    if len(clip['frames']) < t_clip_length:
        return False

    return True

def visualize_dataset():

    from shutil import copyfile

    output_root = 'vis/clips_v1.2'
    io.mkdir_if_missing(output_root)

    clip_lst = io.load_json('datasets/youtube_face/Label/youtube_face_clip_id0.5.json')
    image_root = 'datasets/youtube_face/Images'

    clip_lst = [c for c in clip_lst if _validate_clip(c, t_artifact = 0.2, t_clip_length = 10)]

    for c_idx, c in enumerate(clip_lst):

        if _validate_clip(c, t_artifact = 0.2, t_clip_length = 10):

            io.mkdir_if_missing(os.path.join(output_root, c['identity']))

            output_path = os.path.join(output_root, c['identity'], c['id'])
            io.mkdir_if_missing(output_path)


            for f in c['frames']:

                tar_fn = os.path.join(output_path, os.path.basename(f['image']))
                if not os.path.isfile(tar_fn):
                    copyfile(f['image'], tar_fn)

            clip_info = []
            clip_info.append('clip_id: %s' % c['id'])
            clip_info.append('age: %d' % c['age'])
            clip_info.append('identity: %s' % c['identity'])
            clip_info.append('1st frame: %s' %c['frames'][0]['image'])
            io.save_str_list(clip_info, os.path.join(output_path, 'info.txt'))

            print('visualizing clips: %d / %d' % (c_idx, len(clip_lst)))



if __name__ == '__main__':


    # organize_label_morph()

    # organize_label()

    # organize_clip_label()


    # age_hist_morph()

    # age_hist('all')

    # clip_length_distribution_raw('all')

    # detect_artifact()

    # serialize_clip_lst()

    # add_identity()

    # image_number_per_identity()

    # visualize_dataset()

    clip_length_distribution()

    
    pass
