
from __future__ import print_function, division

import sys
import os
import util.io as io
import util.image as image
import numpy as np


def video_analyze():

    import csv

    
    # load experiment results
    print('loading experiment results ...')

    #======== attribute result  ============
    va_split = io.load_json('datasets/video_age/Labels/v2.0_split_0-70.json')
    id_lst = va_split['test']
    print('sample number: %d' % len(id_lst))

    age_model_id = 'joint_va_3.0.1'
    pose_model_id = 'joint_va_3.0.1' # pose_4.0.2n
    attr_model_id = 'joint_va_3.0.1' # attr_1.0.3

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



if __name__ == '__main__':

    video_analyze()