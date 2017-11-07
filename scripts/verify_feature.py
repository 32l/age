from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import argparse
import numpy as np
import os

import util.io as io
import modules.dataset as dataset
import modules.misc as misc
import modules.gan_model as gan_model
import modules.decoder_model as decoder_model




parser = argparse.ArgumentParser()

parser.add_argument('command', type = str, default = 'verify',
    choices = ['verify'])

parser.add_argument('--id', type = str, default = 'default',
    help = 'model id')

parser.add_argument('--decoder_id', type = str, default = 'dcd_3.2',
    help = 'id of decoder model')

parser.add_argument('--gpu_id', type = int, default = [0], nargs = '*',
    help = 'GPU device id used for testing')


opts = parser.parse_known_args()[0]


# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opts.gpu_id])




def verify_gan2_model(opts):

    # load gan model
    if not opts.id.endswith('.pth'):
        fn = 'models/%s/final.pth' % opts.id
    else:
        fn = opts.id
        opts.id = opts.id.split('/')[-2]

    model = gan_model.GANModel(fn = fn, load_weight = False)
    model.load_model(fn, modules = ['cnn', 'G_net'])
    model.cuda().eval()

    # load decoder
    decoder = decoder_model.DecoderModel(fn = 'models/%s/best.pth' % opts.decoder_id).cuda().eval()

    # load dataset
    dset = dataset.load_video_age_dataset(version = '2.0', subset = 'test',
        crop_size = 128, age_rng = [model.opts.min_age, model.opts.max_age], max_len = 2)

    # number of samples from each group (<30, 30-40, 40-55, >55)
    num_per_group = 4
    num_sample = 4 * num_per_group

    # select sample
    index_lst = []
    index_lst += [idx for idx, a in enumerate(dset.age_lst) if a['age'] < 30][0:num_per_group]
    index_lst += [idx for idx, a in enumerate(dset.age_lst) if 30 <= a['age'] < 40][0:num_per_group]
    index_lst += [idx for idx, a in enumerate(dset.age_lst) if 40 <= a['age'] < 55][0:num_per_group]
    index_lst += [idx for idx, a in enumerate(dset.age_lst) if 55 <= a['age']][0:num_per_group]

    dset.id_lst = [dset.id_lst[idx] for idx in index_lst]
    dset.age_lst = [dset.age_lst[idx] for idx in index_lst]
    dset.video_lst = [dset.video_lst[idx] for idx in index_lst]

    loader = torch.utils.data.DataLoader(dset, batch_size = len(dset), shuffle = False)

    
    # verify sample
    vis_dir = os.path.join('models', opts.id, 'verify_gan')
    io.mkdir_if_missing(vis_dir)

    for idx, data in enumerate(loader):

        img_pair, seq_len, age, _ = data

        img_pair = Variable(img_pair, volatile = True).cuda()
        seq_len = Variable(seq_len, volatile = True).cuda()

        img_in = F.avg_pool2d(img_pair[:,0], kernel_size = 2)
        img_real = F.avg_pool2d(img_pair[:,1], kernel_size = 2)

        _, _, feat_pair = model.forward_video(img_pair, seq_len)
        
        feat_in, feat_real = feat_pair[:,0], feat_pair[:,1]
        noise = Variable(torch.FloatTensor(feat_in.size(0), model.opts.noise_dim).normal_(0, 1)).cuda()
        feat_res = model.G_net(feat_in, noise)
        feat_fake = feat_in + feat_res

        rand_res = Variable(torch.zeros(feat_res.size()).normal_(0, 1)).cuda() * feat_res.std(dim = 0, keepdim = True) + feat_res.mean(dim = 0)
        feat_rand = feat_in + rand_res


        dcd_img_in,_ = decoder(feat_in.contiguous())
        dcd_img_real,_ = decoder(feat_real.contiguous())
        dcd_img_fake,_  = decoder(feat_fake.contiguous())
        dcd_img_rand,_  = decoder(feat_rand.contiguous())


        # (6, 16, 3, 112, 112)
        out_img = torch.stack([img_in, img_real, dcd_img_in, dcd_img_real, dcd_img_fake, dcd_img_rand])

        # (6+6, 8, 3, 112, 112)
        out_img = torch.cat((out_img[:,0:(num_sample//2)], out_img[:,(num_sample//2)::]), dim = 0)

        # (12*8, 3, 112, 112)
        out_img = out_img.contiguous().view(-1, 3, 112, 112)


        fn_vis = os.path.join(vis_dir, '%d.png' % idx)
        torchvision.utils.save_image(out_img.cpu().data, fn_vis, normalize = True)




if __name__ == '__main__':

    cmd = opts.command

    if cmd == 'verify':
        verify_gan2_model(opts)


