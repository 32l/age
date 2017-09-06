from __future__ import division, print_function

import caffe
import util.io as io
import util.image as image
import modules.misc as misc


def test_imdb_wiki_model():

    sample_lst_fn = 'datasets/IMDB-WIKI/Annotations/imdb_wiki_good_test.json'
    img_root = 'datasets/IMDB-WIKI/Images'
    batch_size = 128
    num_batch = 10
    gpu_id = 0

    fn_model = 'datasets/IMDB-WIKI/caffe_models/age.prototxt'
    fn_weight = 'datasets/IMDB-WIKI/caffe_models/dex_imdb_wiki.caffemodel'
    imagenet_mean = [[[104, 117, 123]]]


    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    model = caffemodel(fn_model, fn_weight, caffe.TEST)

    sample_lst = io.load_json(sample_lst_fn)

    if num_batch == 0:
        num_batch = int(np.ceil(len(sample_lst) / batch_size))

    

