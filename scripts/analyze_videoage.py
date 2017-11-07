
from __future__ import print_function, division

import torch
from torch.autograd import Variable

import modules.joint_model as joint_model
import sys
import os
import util.io as io
import util.image as image
import numpy as np


def corr_analyze(age_model_id, pose_model_id = None, attr_model_id = None):

    import csv

    
    # load experiment results
    print('loading experiment results ...')

    #======== attribute result  ============
    va_split = io.load_json('datasets/video_age/Labels/v2.0_split_0-70.json')
    id_lst = va_split['test']
    print('sample number: %d' % len(id_lst))

    # age_model_id = 'joint_va_3.0.1'
    # pose_model_id = 'joint_va_3.0.1' # pose_4.0.2n
    # attr_model_id = 'joint_va_3.0.1' # attr_1.0.3

    if pose_model_id is None:
        pose_model_id = age_model_id
    if attr_model_id is None:
        attr_model_id = age_model_id

    #======== age result  ============
    # fn_age = 'output/video_analysis/video_age_v2.0_detect.pkl'
    fn_age = 'models/%s/video_age_v2.0_test_rst.pkl' % age_model_id
    rst_age = io.load_data(fn_age)
    
    #======== pose result  ============
    # fn_pose = 'output/video_analysis/video_age_v2.0_detect.pkl'
    fn_pose = 'models/%s/video_age_v2.0_test_rst.pkl' % pose_model_id
    rst_pose = io.load_data(fn_pose)

    #======== attribute result  ============
    # fn_attr = 'output/video_analysis/video_age_v2.0_detect.pkl'
    fn_attr = 'models/%s/video_age_v2.0_test_rst.pkl' % attr_model_id
    rst_attr = io.load_data(fn_attr)

    output_fn = 'va_2.0_[%s][%s][%s].csv' % (age_model_id, pose_model_id, attr_model_id)



    # load attribute list
    attr_lst = io.load_str_list('datasets/CelebA/Label/attr_name_lst.txt')
    # attr_lst = io.load_str_list('external/st_SDK_info/attr_name_lst.txt')


    assert len(attr_lst) == len(rst_attr.values()[0]['attr'][0]), \
        '%d vs %d' % (len(attr_lst), len(rst_attr.values()[0]['attr'][0]))

    pose_lst = ['yaw', 'pitch']
    assert len(pose_lst) == len(rst_pose.values()[0]['pose'][0])

    item_lst = ['age'] + pose_lst + attr_lst + ['random']

    # analyze
    print('analyzing ...')

    data_mean = []
    data_var  = []
    data_cov = []
    data_cov1 = []
    data_rng = []

    for idx, s_id in enumerate(id_lst):

        seq_age = rst_age[s_id]['age'] # list [seq_len]
        seq_pose = rst_pose[s_id]['pose'] # list [seq_len, pose_dim]
        seq_attr = rst_attr[s_id]['attr'] # list [seq_len, num_attr]


        seq_len = len(seq_age)

        seq_age = np.array(seq_age).reshape(seq_len, 1)
        seq_pose = np.array(seq_pose) / np.pi * 180.
        # seq_pose = np.array(seq_pose)
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

    with open(os.path.join(output_dir, output_fn), 'wb') as f:
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


        # output info
        csv_writer.writerow(['fn_age', fn_age])
        csv_writer.writerow(['fn_pose', fn_pose])
        csv_writer.writerow(['fn_attr', fn_attr])


def feat_analyze(model_id):

    # config
    num_sample = 20
    num_corr_feat = 20
    skip_top_sample = 0

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    print('load model...')
    fn_model = os.path.join('models', model_id, 'best.pth')
    model = joint_model.JointModel(opts = None, fn = fn_model, fn_cnn = None)
    fc_w_age = model.age_cls.embed.weight.data.numpy()
    corr_feat_idx = np.abs(fc_w_age).sum(axis = 0).argsort()[::-1][0:num_corr_feat]

    print('load result...')
    fn_rst = os.path.join('models', model_id, 'video_age_v2.0_test_rst.pkl')
    rst = io.load_data(fn_rst)


    print('load feature...')
    fn_feat = os.path.join('models', model_id, 'video_age_v2.0_feat.pkl')
    feat_dict = io.load_data(fn_feat)


    # search large difference of age estimation result between 2 frames
    age_diff_lst = []
    for s_id, r in rst.iteritems():
        age = np.array(r['age'])
        age_diff = np.abs(age[0:-1] - age[1::])
        
        idx = age_diff.argmax()

        age_diff_lst.append((s_id, age_diff.max(), idx, age[idx], age[idx+1]))


    age_diff_lst.sort(key = lambda x:x[1], reverse = True)
    age_diff = age_diff_lst[skip_top_sample:skip_top_sample+num_sample]

    
    # feature
    feat_out = []

    for i, s in enumerate(age_diff):
        s_id = s[0]
        t = s[2]
        feat = feat_dict[s_id]
        
        f = feat['feat'][t]
        f_delta = feat['feat_delta'][t]
        f_diff = feat['feat'][t+1] - f

        feat_out.append({
            'feat': f,
            'feat_delta': f_delta,
            'feat_diff': f_diff,
            'feat_corr': f[corr_feat_idx],
            'feat_delta_corr': f_delta[corr_feat_idx],
            'feat_diff_corr': f_diff[corr_feat_idx],
            'age': [s[3], s[4]],
            't': [s[2], s[2]+1],
            'id': s_id,
            })


    for feat_info in feat_out:
        print(feat_info['id'])
    exit(0)
    
    output_dir = os.path.join('output/video_age_feat_analysis/%s' % model_id)
    io.mkdir_if_missing(output_dir)

    va_video = io.load_json('datasets/video_age/Labels/v2.0_video.json')
    
    
    for i, feat_info in enumerate(feat_out):
        s_id = feat_info['id']
        t1, t2 = feat_info['t']
        age1, age2 = feat_info['age']

        fig = plt.figure(figsize = (30,4.8))
        
        # frame t1
        ax = fig.add_subplot(1, 8, 1)
        
        img = mpimg.imread(va_video[s_id]['frames'][t1]['image'])
        ax.imshow(img)
        ax.set_xlabel('%f' % age1)

        # frame t2
        ax = fig.add_subplot(1, 8, 2)

        img = mpimg.imread(va_video[s_id]['frames'][t2]['image'])
        ax.imshow(img)
        ax.set_xlabel('%f' % age2)
        

        # feature
        ax =  fig.add_subplot(1, 8, 3)
        ax.set_ylim([-10,10])
        
        ax.plot(feat_info['feat'])
        ax.set_xlabel('feat')

        # feature delta
        
        ax =  fig.add_subplot(1, 8, 4)
        ax.set_ylim([-10,10])
        
        ax.plot(feat_info['feat_delta'])
        ax.set_xlabel('feat_delta')

        # feature
        
        ax =  fig.add_subplot(1, 8, 5)
        ax.set_ylim([-10,10])
        
        ax.plot(feat_info['feat_diff'])
        ax.set_xlabel('feat_diff')

        # feature
        
        ax =  fig.add_subplot(1, 8, 6)
        ax.set_ylim([-10,10])
        
        ax.plot(feat_info['feat_corr'])
        ax.set_xlabel('feat')

        # feature delta
        
        ax =  fig.add_subplot(1, 8, 7)
        ax.set_ylim([-10,10])
        
        ax.plot(feat_info['feat_delta_corr'])
        ax.set_xlabel('feat_delta')

        # feature
        
        ax =  fig.add_subplot(1, 8, 8)
        ax.set_ylim([-10,10])
        
        ax.plot(feat_info['feat_diff_corr'])
        ax.set_xlabel('feat_diff')

        output_fn = os.path.join(output_dir, 'feat_%d.jpg' % i)
        fig.savefig(output_fn)


def feat_diff_analyze(model_id):

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    # from tsne import bh_sne

    output_dir = os.path.join('models', model_id, 'cnn_feat')

    pca = None
    for name in ['train', 'test']:
        rst = io.load_data(os.path.join(output_dir, name + '.pkl'))
        feat = rst['feat']
        age = rst['age']

        # fd: feature diff
        feat_1, feat_2 = feat[:,0,:], feat[:,1,:]
        fd = feat_1 - feat_2

        # ad: age diff
        ad = age[:,0] - age[:,1]
        
        ad_norm = (ad / ad.mean() * 128).astype(np.int)
        ad_norm[ad_norm>255] = 255

        ### feat_diff distribution
        # pca
        if pca is None:
            pca = PCA(n_components = 2)
            pca.fit(feat_1)
        fd_pca = pca.transform(fd)

        plt.figure()
        plt.scatter(fd_pca[:,0], fd_pca[:,1], c = ad_norm, cmap = plt.get_cmap('plasma'),s = 2)
        plt.savefig(os.path.join(output_dir, '%s-featDiff-ageDiff-pca.jpg' % name))

        # # tsne
        # feat_tsne = bh_sne(np.concatenate((feat_1, feat_2), axis = 0).astype(np.float64))
        # N = feat_1.shape[0]
        # fd_tsne = feat_tsne[0:N] - feat_tsne[N::]

        # plt.figure()
        # plt.scatter(fd_tsne[:,0], fd_tsne[:,1], c = ad_norm, cmap = plt.get_cmap('plasma'), s = 2)
        # plt.savefig(os.path.join(output_dir, 'featDiff-ageDiff-tsne.jpg'))


def feat_res_analyze(model_id):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    import modules.gan_model as gan

    # use noise
    res_noise = False

    # load model
    model = gan.GANModel(fn = os.path.join('models', model_id, 'final.pth'))
    model.cuda()
    model.eval()

    # load cnn feature
    output_dir = os.path.join('models', model_id, 'show_feat')
    io.mkdir_if_missing(output_dir)

    rsts = {}
    for name in ['train', 'test']:
        rsts[name] = io.load_data('models/gan2_pre_1.6/cnn_feat/%s.pkl' % name)

    pca = PCA(n_components = 2)
    pca.fit(rsts['train']['feat'][:,0,:])

    for name in ['train', 'test']:

        feat = rsts[name]['feat']
        age = rsts[name]['age']

        feat_1, feat_2 = feat[:,0,:], feat[:,1,:]
        fd = feat_1 - feat_2

        # compute G_net output
        bsz = 128
        N = feat.shape[0]

        if res_noise:
            mean = fd.mean(axis = 0, keepdims = True)
            std = fd.std(axis = 0, keepdims = True)
            fr = np.random.randn(*fd.shape) * std + mean

        else:
            fr = []
            for s in xrange(0,N,bsz):
                feat_in = feat_1[s:(s+bsz), :]
                feat_in = Variable(torch.FloatTensor(feat_in), volatile = True).cuda()
                noise = Variable(torch.FloatTensor(feat_in.size(0), model.opts.noise_dim).normal_(0, 1)).cuda()
                feat_res = model.G_net(feat_in, noise)

                fr.append(feat_res.data.cpu())
                print('compute feat res [%d / %d]' % (s, N))

            fr = torch.cat(fr, dim = 0).numpy()

        # compute PCA
        fd_pca = pca.transform(fd)
        fr_pca = pca.transform(fr)

        P = np.concatenate((fd_pca, fr_pca), axis = 0)
        C = ['g'] * N + ['r'] * N

        plt.figure()
        plt.scatter(P[:,0], P[:,1], c = C,s = 2)
        
        if res_noise:
            plt.savefig(os.path.join(output_dir, 'featResNoise-pca-%s.jpg'%name))
        else:
            plt.savefig(os.path.join(output_dir, 'featRes-pca-%s.jpg'%name))


if __name__ == '__main__':

    # corr_analyze(*sys.argv[1::])
    # feat_analyze(*sys.argv[1::])
    # feat_diff_analyze(*sys.argv[1::])
    # feat_res_analyze(*sys.argv[1::])


    ## show feat res using PCA
    import shutil
    # model_lst = ['7.%d' % i for i in range(6)] + \
    #             ['6.%d' % i for i in range(13)] + \
    #             ['5.%d' % i for i in range(12)] + \
    #             ['4.%d' % i for i in range(8)] + \
    #             ['3.%d' % i for i in range(16)] + \
    #             ['2.%d' % i for i in range(16)] + \
    #             ['6.9.%d' % i for i in range(1,5)] + \
    #             ['6.12.%d' % i for i in range(1,5)]

    # model_lst = ['6.13.%d' % i for i in range(5)] +\
    #             ['6.14.%d' % i for i in range(5)] + ['6.13', '6.14']


    for s in model_lst:
        try:
            model_id = 'gan2_%s' % s

            
            feat_res_analyze(model_id)
            for name in ['train', 'test']:
                src_file = 'models/%s/show_feat/featRes-pca-%s.jpg' % (model_id, name)
                dst_file = 'output/feat_res/%s_featRes-pca-%s.jpg' % (model_id, name)
                shutil.copyfile(src_file, dst_file)
            print(model_id)

        except:
            pass


