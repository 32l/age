from __future__ import division, print_function

import util.io as io
import util.image as image
import modules.misc as misc
import numpy as np
import os


def test_bmvc_model():

    # config
    fn_model    = 'external/AgeEstimation0630/age_model/fix_weight/deploy.prototxt'
    fn_weight   = 'external/AgeEstimation0630/age_model/fix_weight/fix_weight_fold_0.caffemodel'
    sample_lst_fn = 'datasets/megaAge/Label/megaage_train.json'
    img_root = 'datasets/megaAge'

    gpu_id      = 1
    batch_size  = 128

    crop_size   = 110
    final_size  = 256
    y_offset    = 15

    num_sample  = 1000
    
    # load model
    import caffe
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    model = caffe.Net(fn_model, fn_weight, caffe.TEST)

    # load sample list
    sample_lst = io.load_json(sample_lst_fn)[17146::]
    if num_sample < 0:
        num_sample = len(sample_lst)

    age_gt = np.array([s['age'] for s in sample_lst], dtype = np.float32)[0:num_sample]
    age_est = np.zeros(num_sample, dtype = np.float32)

    # age estimation
    # age_est = np.ones(num_sample, dtype = np.float32) * age_gt.mean()

    for idx in xrange(0, num_sample, batch_size):

        print('processing sample %d / %d' % (idx, num_sample))

        bsz = min(batch_size, num_sample - idx)
        batch = np.zeros((bsz, final_size, final_size, 3), dtype = np.float32)

        for i in xrange(bsz):
            img = image.imread(os.path.join(img_root, sample_lst[i+idx]['image']))

            # Todo: add image preprocessing
            h, w = img.shape[0:2]
            img = image.crop(img, bbox = [(w - crop_size)/2, (h - crop_size)/2 + y_offset, (w + crop_size)/2, (h + crop_size)/2 + y_offset])
            img = image.resize(img, (final_size, final_size))
            
            batch[i,:,:,:] = img

        batch = batch * 3.2 / 255 - 1.6
        batch = np.transpose(batch, axes = (0, 3, 1, 2))


        # forward
        model.blobs['data'].reshape(*batch.shape)
        model.blobs['data'].data[...] = batch
        output = model.forward()

        # output['normalized_age'].shape = (bsz, 70)
        age_est[idx:(idx + bsz)] = output['normalized_age'].sum(axis = 1)

    # compute metrics
    crit_acc = misc.Cumulative_Accuracy()
    crit_acc.add(age_est, age_gt)

    # print(age_est[0:10])
    # print(age_gt[0:10])
    # for i in range(10):
    #     print(sample_lst[i]['image'])

    print('MAE: %.2f   CA(3): %.2f   CA(5): %.2f   CA(10): %.2f' % 
        (crit_acc.mae(), crit_acc.ca(3), crit_acc.ca(5), crit_acc.ca(10)))


if __name__ == '__main__':
    test_bmvc_model()