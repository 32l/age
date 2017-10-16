from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import util.io as io
from util.pavi import PaviClient
import dataset
import misc
import opt_parser_gan as opt_parser

import os
import sys
import numpy as np
from collections import OrderedDict
import time


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    elif classname.fine('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.fill_(0.0)


class GANModel(nn.Module):

    def _update_opts(self, opts):
        return opts

    def __init__(self, opts = None, fn = None, fn_cnn = None):
        '''
        Create an age model. Input should be one of following combinations:

            opts:   
                Create model architecture by input options. cnn is initiated by weights pretrained on ImageNet,
                cls is initiated by random weights.

            opts + fn_cnn:
                Create model architecture by input options. cnn is loaded from fn_cnn, cls is initiated by
                random weights.

            fn:
                Load model architecture and all model weights from fn.
                Note that fn will omit opts,

        '''

        assert (opts or fn), 'Error: either "opts" or "fn" should be provided'

        print('[GAN.init] fn: %s' % fn)
        print('[GAN.init] fn_cnn: %s' % fn_cnn)
        # print('[JointModel.init] opts: %s' % opts)


        super(GANModel, self).__init__()

        if opts is None:
            opts = torch.load(fn, map_location=lambda storage, loc: storage)['opts']
            opts = self._update_opts(opts)

        self.opts = opts

        ## create model
        # cnn
        if opts.cnn == 'resnet18':
            net = torchvision.models.resnet18(pretrained = True)
            cnn_layers = net._modules
            cnn_layers.popitem() # remove last fc layer
            self.cnn = nn.Sequential(cnn_layers)
            self.cnn_feat_size = 512

        elif opts.cnn == 'resnet50':
            net = torchvision.models.resnet50(pretrained = True)
            cnn_layers = net._modules
            cnn_layers.popitem() # remove last fc layer
            self.cnn = nn.Sequential(cnn_layers)
            self.cnn_feat_size = 2048

        elif opts.cnn == 'vgg16':
            net = torchvision.models.vgg16(pretrained = True)
            cnn_layers = net.features._modules
            # replace the last maxpooling layer (kernel_sz = 2, stride = 2) with a more spares one.
            cnn_layers['30'] = nn.MaxPool2d(kernel_size = (7, 7), stride = (7, 7))
            self.cnn = nn.Sequential(cnn_layers)
            self.cnn_feat_size = 2048 #(512 * 2 * 2)

        else:
            raise Exception('invalid cnn type %s' % opts.cnn)



        # age classifier

        if opts.cls_type == 'oh':
            output_size = opts.max_age - opts.min_age
        elif opts.cls_type == 'dex':
            output_size = opts.max_age - opts.min_age + 1
        elif opts.cls_type == 'reg':
            output_size = 2
        else:
            raise Exception('invalid age classifier type: %s' % opts.cls_type)

        self.age_cls = nn.Sequential(OrderedDict([
                ('fc0', nn.Linear(self.feat_size, opts.age_fc_size, bias = True)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(p = opts.dropout)),
                ('cls', nn.Linear(opts.age_fc_size, output_size, bias = True))
                ]))


        # GAN
        # generator

        self.G_net = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(self.feat_size + opts.noise_dim, opts.G_h0)),
            ('bn0', nn.BatchNorm1d(opts.G_h0)),
            ('elu0', nn.ELU()),
            ('dropout0', nn.Dropout(p = opts.gan_dropout)),
            ('fc1', nn.Linear(self.G_h0, self.feat_size))
            ]))


        # discriminator
        self.D_net = nn.Sequential(OrderedDict([
            ('relu', nn.ReLU()),
            ('fc0', nn.Linear(self.feat_size, 1, bias = False)),
            ('sigmoid', nn.Sigmoid())
            ]))


        
        # init weight
        if fn:
            print('[GANModel.init] loading weights from %s' % fn)
            model_info = torch.load(fn, map_location=lambda storage, loc: storage)
            self.cnn.load_state_dict(model_info['cnn_state_dict'])
            self.age_cls.load_state_dict(model_info['age_cls_state_dict'])
            self.G_net.load_state_dict(model_info['G_net_state_dict'])
            self.D_net.load_state_dict(model_info['D_net_state_dict'])

        elif fn_cnn:
            print('[GANModel.init loading CNN weights from %s' % fn_cnn)
            model_info = torch.load(fn, map_location=lambda storage, loc: storage)
            self.cnn.load_state_dict(model_info['cnn_state_dict'])

            for m in [self.age_cls, self.G_net, self.D_net]:
                m.apply(weights_init)
        else:

            for m in [self.age_cls, self.G_net, self.D_net]:
                m.apply(weights_init)


    def _get_state_dict(self, model = None):
        
        if model is None:
            model = self

        state_dict = OrderedDict()
        for p_name, p in model.state_dict().iteritems():
            # remove the prefix "module.", which is added when using dataparaller for multi-gpu training
            p_name = p_name.replace('module.', '')
            state_dict[p_name] = p.cpu()

        return state_dict


    def save_model(self, fn):

        model_info = {
            'opts': self.opts,
            'cnn_state_dict': self._get_state_dict(self.cnn),
            'age_cls_state_dict': self._get_state_dict(self.age_cls),
            'G_net_state_dict': self._get_state_dict(self.G_net),
            'D_net_state_dict': self._get_state_dict(self.D_net)
        }

        torch.save(model_info, fn)



    def _forward_age_cls(self, feat):

        fc_out = self.age_cls(F.relu(feat))

        if self.opts.cls_type == 'dex':
            # Deep EXpectation
            age_scale = np.arange(self.opts.min_age, self.opts.max_age + 1, 1.0)
            age_scale = Variable(fc_out.data.new(age_scale)).unsqueeze(1)

            age_out = torch.matmul(F.softmax(fc_out), age_scalei).view(-1)
            

        elif self.opts.cls_type == 'oh':
            # Ordinal Hyperplane
            fc_out = F.sigmoid(fc_out)
            age_out = fc_out.sum(dim = 1) + self.opts.min_age

        elif self.opts.cls_type == 'reg':
            # Regression
            age_out = fc_out.view(-1)
            age_out = age_out + self.opts.min_age

        return age_out, fc_out


    def forward(self, img):
        '''
        forward process of age model

        Input:
            img: (bsz, 3, 224, 224). Image data mini-batch
        Output:
            age_out: (bsz, 1). Predicted age.
            fc_out: (bsz, fc_age). Output of the last fc-layer

        '''

        feat = self.cnn(img)
        feat = feat.view(feat.size(0), -1)

        age_out, fc_out = self._forward_age_cls(feat)

        return age_out, fc_out

    def forward_video(self, img_seq, seq_len):
        '''
        Forward video clips
        Input: 
            img_seq: (bsz, max_len, 3, 224, 224). Video data mini-batch
            seq_len: (bsz,). Length of each video clip.
        Output:
            age_out: (bsz, max_len). Predicted age
            fc_out:  (bsz, max_len, fc_size)

        '''

        bsz, max_len = img_seq.size()[0:2]

        img_seq = img_seq.view(bsz * max_len, img_seq.size(2), img_seq.size(3), img_seq.size(4))

        age_out, fc_out = self.forward(img_seq)

        age_out = age_out.view(bsz, max_len)
        fc_out = fc_out.view(bsz, max_len, -1)

        return age_out, fc_out




def pretrain(model, train_opts):

    if not train_opts.id.startswith('gan_'):
        train_opts.id = 'gan_' + train_opts.id

    opts_str = opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParaleel(model.cnn)
    model.cuda()


    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = train_opts.video_max_len)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age])

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = 32, 
        num_workers = 4, pin_memory = True)


    ### create optimizer
    if train_opts.optim == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.cnn.parameters(), 'lr_mult': 1.0},
            {'params': model.age_cls.parameters(), 'lr_mult': train_opts.age_cls_lr_mult}
            ], lr = train_opts.lr, weight_decay = train_opts.weight_decay, momentum = train_opts.momentum)

    elif train_opts.optim == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.cnn.parameters(), 'lr_mult': 1.0},
            {'params': model.age_cls.parameters(), 'lr_mult': train_opts.age_cls_lr_mult}
            ], lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
            eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    ### loss function
    if model.opts.cls_type == 'dex':
        crit_age = nn.CrossEntropyLoss(ignore_index = -1)
    elif model.opts.cls_type == 'oh':
        crit_age = misc.Ordinal_Hyperplane_Loss(relaxation = model.opts.oh_relaxation, ignore_index = -1)
    elif model.opts.cls_type == 'reg':
        crit_age = nn.MSELoss()

    crit_age = misc.Smooth_Loss(misc.Video_Loss(crit_age))
    meas_age = misc.Video_Age_Analysis()


    ### output information
    # json_log
    output_dir = os.path.join('models', train_opts.id)
    io.mkdir_if_missing(output_dir)
    fn_info = os.path.join(output_dir, 'info.json')

    info = {
        'opts': vars(model.opts),
        'train_opts': vars(train_opts),
        'train_history': [],
        'test_history': [],
    }

    def _snapshot(epoch):
        print('saving checkpoint to %s\t' % output_dir)
        model.save_model(os.path.join(output_dir, '%s.pth' % epoch))
        io.save_json(info, fn_info)

    # text_log
    fout = open(os.path.join(output_dir, 'log.txt'), 'w')
    print(opts_str, file = fout)

    # pavi_log
    if train_opts.pavi == 1:
        pavi = PaviClient(username = 'ly015', password = '123456')
        pavi.connect(model_name = train_opts.id, info = {'session_text': opts_str})


    # save checkpoint if getting a best performance
    checkbest_name = 'mae'
    checkbest_value = sys.float_info.max
    checkbest_epoch = -1


    ### main training loop
    epoch = 0

    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg['lr_mult']

        # train one epoch
        for batch_size, age_data in enumerate(train_loader):

            optimizer.zero_grad()

            img_seq, seq_len, age_gt, age_std = data
            img_seq = Variable(img_seq).cuda()
            seq_len = Variable(seq_len).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_std = age_std.float()

            age_label = age_gt.round().long() - model.opts.min_age

            # forward and backward
            age_out, fc_out = model.forward_video(img_seq, seq_len)

            loss = crit_age(fc_out, age_label, seq_len)
            meas_age.add(age_out, age_gt, seq_len, age_std)

            loss.backward()


            # optimize
            optimizer.step()

            # display
            if batch_idx % train_opts.display_interval == 0:
                loss_smooth = crit_age.smooth_loss()
                mae_smooth = meas_age.mae()

                crit_age.clear()
                meas_age.clear()

                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)]   LR: %.3e   Loss: %.6f   Mae: %.2f' %\
                        (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size,
                        len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        lr, loss_smooth, mae_smooth)

                print(log) # to screen
                print(log, file = fout) # to log file

                iteration = batch_idx + epoch * len(train_loader)
                info['train_history'].append({
                    'iteration': iteration,
                    'epoch': epoch, 
                    'loss': loss_smooth, 
                    'mae': mae_smooth})

                if train_opts.pavi == 1:
                    pavi_outputs = {
                        'loss_age': loss_smooth,
                        'mae_age_upper': mae_smooth
                    }
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)

            # update epoch index
            epoch += 1

            # test
            for batch_idx, data in enumerate(test_loader):

                img_seq, seq_len, age_gt, age_std = data
                img_seq = Variable(img_seq).cuda()
                seq_len = Variable(seq_len).cuda()
                age_gt = Variable(age_gt.float()).cuda()
                age_std = age_std.float()

                age_label = age_gt.round().long() - model.opts.min_age

                # forward
                age_out, fc_out = model.forward_video(img_seq, seq_len)

                loss = crit_age(fc_out, age_label, seq_len)
                meas_age.add(age_out, age_gt, seq_len, age_std)

                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()

                if batch_idx + 1 == test_iter:
                    break
            print('')

            # display
            loss = crit_age.smooth_loss()
            mae = meas_age.mae()
            ca3 = meas_age.ca(3)
            ca5 = meas_age.ca(5)
            ca10 = meas_age.ca(10)
            lap_err = meas_age.lap_err()
            der = meas_age.stable_der()
            rng = meas_age.stable_range()

            crit_age.clear()
            meas_age.clear()

            log = '[%s] [%s] Test Epoch %d   Loss: %.6f   Mae: %.2f\n\tCA(3): %.2f   CA(5): %.2f   CA(10): %.2f   LAP: %.4f\n\tDer: %f   Range: %f' % \
                    (time.ctime(), train_opts.id, epoch, loss, mae, ca3, ca5, ca10, lap_err, der, rng)

            print(log)
            print(log, file = fout)

            iteration = epoch * len(train_loader)

            info['test_history'].append({
                    'iteration': iteration,
                    'epoch': epoch, 
                    'loss': loss, 
                    'mae': mae,
                    'ca3': ca3,
                    'ca5': ca5,
                    'ca10': ca10,
                    'lap_err': lap_err,
                    'der': der,
                    'range': rng
                    })

            if train_opts.pavi == 1:
                pavi_outputs = {
                    'loss_age': loss,
                    'mae_age_upper': mae,
                    'der_age_upper': der
                }
                pavi.log(phase = 'test', iter_num = iteration, outputs = pavi_outputs)

            # snapshot best
            if info['test_history'][-1][checkbest_name] < checkbest_value:
                checkbest_value = info['test_history'][-1][checkbest_name]
                checkbest_epoch = epoch
                _snapshot('best')

        # snapshot
        if train_opts.snapshot_interval > 0 and epoch % train_opts.snapshot_interval == 0:
            _snapshot(epoch)

    # final snapshot
    _snapshot(epoch = 'final')

    log = 'best performance: epoch %d' % checkbest_epoch
    print(log)
    print(log, file = fout)
    fout.close()


if __name__ == '__main__':

    command = opt_parser.parse_command()

    if command == 'pretrain':

        model_opts = opt_parser.parse_opts_gan_model()
        train_opts = opt_parser.parse_opts_train()

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        # cnn file
        if not train_opts.pre_id.endswith('.pth'):
            fn_cnn = os.path.join('models', train_opts.pre_id, 'best.pth')
        else:
            fn_cnn = train_opts.pre_id

        model = GANModel(opts = model_opts, fn_cnn = fn_cnn)

        pretrain(model, train_opts)

    else:
        raise Exception('invalid command "%s"' % command)
