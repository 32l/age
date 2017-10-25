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

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


class GANModel(nn.Module):

    def _update_opts(self, opts):
        return opts

    def __init__(self, opts = None, fn = None):
        '''
        Create an age model. Input should be one of following combinations:

            opts:   
                Create model architecture by input options. cnn is initiated by weights pretrained on ImageNet,
                cls is initiated by random weights.

            fn:
                Load model architecture and all model weights from fn.
                Note that fn will omit opts,

        '''

        assert (opts or fn), 'Error: either "opts" or "fn" should be provided'

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
                ('fc0', nn.Linear(self.cnn_feat_size, opts.age_fc_size, bias = True)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(p = opts.dropout)),
                ('cls', nn.Linear(opts.age_fc_size, output_size, bias = True))
                ]))


        # GAN
        # generator
        g_hidden_lst = [self.cnn_feat_size + opts.noise_dim] + opts.G_hidden + [self.cnn_feat_size]
        g_layers = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(g_hidden_lst, g_hidden_lst[1::])):
            g_layers['fc%d'%n] = nn.Linear(dim_in, dim_out, bias = False)
            if n < len(g_hidden_lst) - 2:
                g_layers['bn%d'%n] = nn.BatchNorm1d(dim_out)
                # g_layers['leaky_relu%d'%n] = nn.LeakyReLU(0.2)
                # g_layers['relu%d'%n] = nn.ReLU()
                g_layers['elu%d'%n] = nn.ELU()
        self.G_net = nn.Sequential(g_layers)

        # discriminator
        if opts.D_mode == 'cond':
            d_hidden_lst = [self.cnn_feat_size * 2] + opts.D_hidden +[1]
        else:
            d_hidden_lst = [self.cnn_feat_size] + opts.D_hidden +[1]

        d_layers = OrderedDict()
        # d_layers['relu'] = nn.ReLU()
        for n, (dim_in, dim_out) in enumerate(zip(d_hidden_lst, d_hidden_lst[1::])):
            d_layers['fc%d'%n] = nn.Linear(dim_in, dim_out, bias = False)
            if n < len(d_hidden_lst) - 2:
                if opts.D_bn == 1:
                    d_layers['bn%d'%n] = nn.BatchNorm1d(dim_out)
                d_layers['leaky_relu%d'%n] = nn.LeakyReLU(0.2)
        d_layers['sigmoid'] = nn.Sigmoid()
        self.D_net = nn.Sequential(d_layers)
        
        
        # init weight1
        if fn:
            self.load_model(fn)
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


    def load_model(self, fn, modules = None):

        if modules is None:
            modules = ['cnn', 'age_cls', 'G_net', 'D_net']

        model_info = torch.load(fn, map_location=lambda storage, loc: storage)

        for m_name in modules:
            self.__getattr__(m_name).load_state_dict(model_info['%s_state_dict' % m_name])
            print('[GANModel.load_model] %s <= %s' % (m_name, fn))



    def _forward_age_cls(self, feat):

        fc_out = self.age_cls(feat)

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
            feat: (bsz, feat_size)

        '''

        feat = self.cnn(img)
        feat = feat.view(feat.size(0), -1)

        age_out, fc_out = self._forward_age_cls(feat)

        return age_out, fc_out, feat

    def forward_video(self, img_seq, seq_len):
        '''
        Forward video clips
        Input: 
            img_seq: (bsz, max_len, 3, 224, 224). Video data mini-batch
            seq_len: (bsz,). Length of each video clip.
        Output:
            age_out: (bsz, max_len). Predicted age
            fc_out:  (bsz, max_len, fc_size)
            feat: (bsz, max_len, feat_size)
        '''

        bsz, max_len = img_seq.size()[0:2]

        img_seq = img_seq.view(bsz * max_len, img_seq.size(2), img_seq.size(3), img_seq.size(4))

        age_out, fc_out, feat = self.forward(img_seq)

        age_out = age_out.view(bsz, max_len)
        fc_out = fc_out.view(bsz, max_len, -1)
        feat = feat.view(bsz, max_len, -1)

        return age_out, fc_out, feat
        
    def forward_video_with_feat_aug(self, img_seq, seq_len, opts):
        '''
        Forward video clips with feature augmentation for each frame. Assume that all
        frames are valid
        Input: 
            img_seq: (bsz, max_len, 3, 224, 224). Video data mini-batch
            seq_len: (bsz,). Length of each video clip.
            opts: (namespace) training options, which contains augmentation options
        Output:
            age_out: (bsz, expanded_len). Predicted age
            fc_out:  (bsz, expanded_len, fc_size)

        '''
        
        ### aug otps
        aug_mode = opts.aug_mode
        aug_rate = opts.aug_rate
        aug_scale = opts.aug_scale
        
        age_out, fc_out, feat = self.forward_video(img_seq, seq_len)
        bsz, org_len, feat_sz = feat.size()
        
        feat_exp = feat.view(bsz*org_len, feat_sz).unsqueeze(dim = 1).expand(aug_rate).contiguous().view(bsz*org_len*aug_rate, feat_sz)
        
        
        if aug_mode == 'gan':
            # use G_net to generate feature residual
            noise = Variable(torch.FloatTensor(feat_exp.size(0), model.opts.noise_dim).normal_(0, 1)).cuda()
            feat_res = self.G_net(torch.cat((feat_exp, noise), dim = 1))
            
        elif aug_mode == 'gaussian':
            # add gaussian noise to original feature
            feat_res = Variable(torch.FloatTensor(feat_exp.size()).normal_(0, aug_scale).cuda())
        
        feat_exp = F.relu(feat_exp + feat_res)
        age_exp, fc_exp = self._forward_age_cls(feat_exp)
        
        age_exp = age_exp.view(bsz, org_len*aug_rate)
        fc_exp = fc_exp.view(bsz, org_len*aug_rate, -1)
        feat_exp = feat.view(bsz, org_len*aug_rate, -1)
        
        age_out = torch.cat((age_out, age_exp), dim = 1)
        fc_out = torch.cat((fc_out, fc_exp), dim = 1)
        feat = torch.cat((feat, feat_exp), dim = 1)
        
        return age_out, fc_out, feat
        
        



def pretrain(model, train_opts):

    if not train_opts.id.startswith('gan_'):
        train_opts.id = 'gan_' + train_opts.id

    opts_str = opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()


    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = train_opts.video_max_len)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age])

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = 128, 
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
    model.eval()
    
    while epoch < train_opts.max_epochs:

        # set model mode
        model.G_net.train()
        model.D_net.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg['lr_mult']

        # train one epoch
        for batch_idx, data in enumerate(train_loader):

            optimizer.zero_grad()

            img_seq, seq_len, age_gt, age_std = data
            img_seq = Variable(img_seq).cuda()
            seq_len = Variable(seq_len).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_std = age_std.float()

            age_label = age_gt.round().long() - model.opts.min_age

            # forward and backward
            age_out, fc_out, _ = model.forward_video(img_seq, seq_len)

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
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test model
            model.eval()

            # clear buffer
            crit_age.clear()
            meas_age.clear()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)    

            
            for batch_idx, data in enumerate(test_loader):

                img_seq, seq_len, age_gt, age_std = data
                img_seq = Variable(img_seq).cuda()
                seq_len = Variable(seq_len).cuda()
                age_gt = Variable(age_gt.float()).cuda()
                age_std = age_std.float()

                age_label = age_gt.round().long() - model.opts.min_age

                # forward
                age_out, fc_out, _ = model.forward_video(img_seq, seq_len)

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


def pretrain_gan(model, train_opts):
    
    if not train_opts.id.startswith('gan_'):
        train_opts.id = 'gan_' + train_opts.id
    
    opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)
    
    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()

    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = 2)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        max_len = 2)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = 16, 
        num_workers = 4, pin_memory = True)

    ### create optimizer
    if train_opts.optim == 'sgd':
        # use SGD
        optimizer_G = torch.optim.SGD(model.G_net.parameters(), lr = train_opts.lr, weight_decay = train_opts.weight_decay,
            momentum = train_opts.momentum)
        optimizer_D = torch.optim.SGD(model.D_net.parameters(), lr = train_opts.lr, weight_decay = train_opts.weight_decay,
            momentum = train_opts.momentum)
            
    elif train_opts.optim == 'adam':
        # use Adam
        optimizer_G = torch.optim.Adam(model.G_net.parameters(), lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
                eps = train_opts.optim_epsilon)

        optimizer_D = torch.optim.Adam(model.D_net.parameters(), lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta),
                eps = train_opts.optim_epsilon)
    
    ### loss function
    crit_G = misc.Smooth_Loss(nn.L1Loss()) # G Loss
    crit_D_R = misc.Smooth_Loss(nn.BCELoss()) # D loss with feat_real
    crit_D_F = misc.Smooth_Loss(nn.BCELoss()) # D loss with feat_fake

    # GAN observer
    meas_D_R = misc.Smooth_Loss(misc.BlankLoss()) # D(feat_real)
    meas_D_F = misc.Smooth_Loss(misc.BlankLoss()) # D(feat_fake)
    meas_acc = misc.Smooth_Loss(misc.BCEAccuracy()) # D classification acc
    
    # Age observer
    meas_age_mae = misc.Smooth_Loss(nn.L1Loss()) # age estimation MAE
    meas_age_diff_real = misc.Smooth_Loss(nn.L1Loss()) # age estimation difference between feat_in and feat_real
    meas_age_diff_fake = misc.Smooth_Loss(nn.L1Loss()) # age estimation difference between feat_in and feat_fake

    # Feature norm observer
    meas_feat_diff_fake = misc.Smooth_Loss(misc.BlankLoss()) # (feat_res norm) / (feat_in norm)
    meas_feat_diff_real = misc.Smooth_Loss(misc.BlankLoss()) # (feat_real-feat_in norm) / (feat_in norm)
    meas_grad_norm = misc.Smooth_Loss(misc.BlankLoss()) # (feat_res grad norm)
    
    ### output information
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
        fn_snapshot = os.path.join(output_dir, '%s.pth' % epoch)
        print('saving checkpoint to %s' % fn_snapshot)
        model.save_model(fn_snapshot)
        io.save_json(info, fn_info)


    # text_log
    fout = open(os.path.join(output_dir, 'log.txt'), 'w')
    print(opts_str, file = fout)

    # pavi_log
    if train_opts.pavi == 1:
        pavi = PaviClient(username = 'ly015', password = '123456')
        pavi.connect(model_name = train_opts.id, info = {'session_text': opts_str})

    
    ### main training loop
    real_label = 1.
    fake_label = 0.
    model.eval()
    
    epoch = 0    
    while epoch < train_opts.G_max_epochs + train_opts.D_max_epochs:
        
        # set model mode
                
        # update learning rate
        if epoch < train_opts.G_max_epochs:
            lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.G_lr_decay))
            optimizer_G.param_groups[0]['lr'] = lr
            optimizer_D.param_groups[0]['lr'] = 0. # fix the parameter of D
            model.G_net.train()
            model.D_net.eval()
        
        else:
            lr = train_opts.lr * (train_opts.lr_decay_rate ** ((epoch - train_opts.G_max_epochs) // train_opts.D_lr_decay))
            optimizer_G.param_groups[0]['lr'] = 0. # fix the parameters of G
            optimizer_D.param_groups[0]['lr'] = lr
            model.G_net.eval()
            model.D_net.train()
        
        # train one epoch
        for batch_idx, data in enumerate(train_loader):
            iteration = batch_idx + epoch*len(train_loader)

            ### extract feature
            img_pair, seq_len, age_gt, _ = data
            img_pair = Variable(img_pair).cuda()
            age_gt = Variable(age_gt.float()).cuda()

            bsz = img_pair.size(0)

            age_out, _, feat = model.forward_video(img_pair, seq_len)
            feat.detach_()
            age_out.detach_()
            
            feat_in = feat[:,0,:]
            feat_real = feat[:,1,:]
            age_in = age_out[:,0]
            age_real = age_out[:,1]

            #### train D_net            
            optimizer_D.zero_grad()

            # train with real
            if model.opts.D_mode == 'cond':
                    out = model.D_net(torch.cat((feat_in, feat_real), dim = 1))
            else:
                out = model.D_net(feat_real)

            label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()
            
            loss_real = crit_D_R(out, label)
            _ = meas_acc(out, label)
            _ = meas_D_R(out, None)
            

            loss_real.backward()

            # train with fake
            noise = Variable(torch.FloatTensor(bsz, model.opts.noise_dim).normal_(0, 1)).cuda()
            feat_res = model.G_net(torch.cat((feat_in, noise), dim = 1))
            feat_fake = F.relu(feat_in + feat_res)
            
            # detach the feat_fake, so that the grad of loss_d will not back propagate to G_net
            if model.opts.D_mode == 'cond':
                out = model.D_net(torch.cat((feat_in, feat_fake.detach()), dim = 1))
            else:
                out = model.D_net(feat_fake.detach())

            label = Variable(torch.FloatTensor(bsz, 1).fill_(fake_label)).cuda()

            loss_fake = crit_D_F(out, label)

            _ = meas_acc(out, label)
            _ = meas_D_F(out, None)

            loss_fake.backward()

            # update D_net
            optimizer_D.step()
            
            ### train G_net
            optimizer_G.zero_grad()
            feat_fake.retain_grad() # grad of feat_fake is quivalent to grad of feat_res
            
            # G loss
            loss_g = crit_G(feat_fake, feat_real)
            loss_g.backward()
            optimizer_G.step()

            # L2 norm
            l2_res = (feat_fake - feat_in).norm(p = 2, dim = 1, keepdim = True)
            l2_diff = (feat_real - feat_in).norm(p = 2, dim = 1, keepdim = True)
            
            ### observations
            age_fake, _ = model._forward_age_cls(feat_fake)
            _ = meas_age_mae(age_in, age_gt)
            _ = meas_age_diff_real(age_real, age_in)
            _ = meas_age_diff_fake(age_fake, age_in)
            
            feat_norm = feat_in.norm(p = 2, dim = 1, keepdim = True)
            _ = meas_feat_diff_fake(l2_res, None)
            _ = meas_feat_diff_real(l2_diff, None)
            _ = meas_grad_norm(feat_fake.grad.norm(p = 2, dim = 1, keepdim = True), None)
            
            ### display
            if batch_idx % train_opts.display_interval == 0:

                loss_g = crit_G.smooth_loss(clear = True)
                loss_d = (crit_D_R.smooth_loss(clear = True) + crit_D_F.smooth_loss(clear = True)) / 2
                D_real = meas_D_R.smooth_loss(clear = True)
                D_fake = meas_D_F.smooth_loss(clear = True)
                D_acc = meas_acc.smooth_loss(clear = True)

                age_mae = meas_age_mae.smooth_loss(clear = True)
                ad_real = meas_age_diff_real.smooth_loss(clear = True)
                ad_fake = meas_age_diff_fake.smooth_loss(clear = True)
                fd_real = meas_feat_diff_real.smooth_loss(clear = True)
                fd_fake = meas_feat_diff_fake.smooth_loss(clear = True)
                grad_g = meas_grad_norm.smooth_loss(clear = True)


                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)] LR: %.3e   Loss_G: %.6f   loss_D: %.6f'% \
                    (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset), 100.*batch_idx / len(train_loader),
                        lr, loss_g, loss_d)
                log += '\n\tD_real: %.6f   D_fake: %.6f   D_acc: %.2f' % (D_real, D_fake, D_acc * 100.)
                log += '\n\t Age MAE: [GT: %.2f   Real: %.2f   Fake: %.2f]' % (age_mae, ad_real, ad_fake)
                log += '\n\t Feat Diff: [Real: %.6f   Fake: %.6f]  Feat Grad: %.6f' % (fd_real, fd_fake, grad_g)

                print(log)
                print(log, file = fout)
                
                info['train_history'].append({
                    'iteration': iteration,
                    'epoch': epoch,
                    'Loss_G': loss_g,
                    'loss_D': loss_d,
                    'D_real': D_real,
                    'D_fake': D_fake,
                    'D_acc': D_acc,
                    'age_mae': age_mae,
                    'ad_real': ad_real,
                    'ad_fake': ad_fake,
                    'fd_real': fd_real,
                    'fd_fake': fd_fake,
                    'grad_g': grad_g
                    })

                if train_opts.pavi == 1:
                    pavi_outputs = {
                        'Loss_G': loss_g,
                        'loss_D': loss_d,
                        'delta': fd_fake,
                        'D_real_upper': D_real,
                        'D_fake_upper': D_fake,
                        'D_acc_upper': D_acc,
                    }
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)

        ### update epoch index
        epoch += 1
        
        ### test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test mode
            model.eval()

            # clear loss buffer
            for lb in [crit_G, crit_D_R, crit_D_F, meas_D_R, meas_D_F, meas_acc,\
                        meas_age_mae, meas_age_diff_real, meas_age_diff_fake,\
                        meas_feat_diff_real, meas_feat_diff_fake, meas_grad_norm]:
                lb.clear()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)

            for batch_idx, data in enumerate(test_loader):


                img_pair, seq_len, age_gt, _ = data
                img_pair = Variable(img_pair).cuda()
                age_gt = Variable(age_gt.float()).cuda()

                bsz = img_pair.size(0) * 2

                age_out, _, feat = model.forward_video(img_pair, seq_len)
                feat.detach_()
                age_out.detach_()

                feat_in = torch.cat((feat[:,0,:], feat[:,1,:]))
                feat_real = torch.cat((feat[:,1,:], feat[:,0,:]))
                age_in = torch.cat((age_out[:,0], age_out[:,1]))
                age_real = torch.cat((age_out[:,1], age_out[:,0]))
                age_gt = torch.cat((age_gt, age_gt))

                noise = Variable(torch.FloatTensor(bsz, model.opts.noise_dim).normal_(0, 1)).cuda()
                feat_res = model.G_net(torch.cat((feat_in, noise), dim = 1))
                feat_fake = F.relu(feat_in + feat_res)

                # forward D real
                if model.opts.D_mode == 'cond':
                    out = model.D_net(torch.cat((feat_in, feat_real), dim = 1))
                else:
                    out = model.D_net(feat_real)

                label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()

                _ = crit_D_R(out, label)
                _ = meas_acc(out, label)
                _ = meas_D_R(out, None)

                # forward D fake
                if model.opts.D_mode == 'cond':
                    out = model.D_net(torch.cat((feat_in, feat_fake), dim = 1))
                else:
                    out = model.D_net(feat_fake)

                label = Variable(torch.FloatTensor(bsz, 1).fill_(fake_label)).cuda()

                _ = crit_D_F(out, label)
                _ = meas_acc(out, label)
                _ = meas_D_F(out, None)
                
                # forward G
                _ = crit_G(feat_fake, feat_real)
                
                # observations
                age_fake, _ = model._forward_age_cls(feat_fake)
                _ = meas_age_mae(age_in, age_gt)
                _ = meas_age_diff_real(age_real, age_in)
                _ = meas_age_diff_fake(age_fake, age_in)

                feat_norm = feat_in.norm(p = 2, dim = 1, keepdim = True)
                _ = meas_feat_diff_fake((feat_fake - feat_in).norm(p = 2, dim = 1, keepdim = True), None)
                _ = meas_feat_diff_real((feat_real - feat_in).norm(p = 2, dim = 1, keepdim = True), None)


                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()

                if batch_idx + 1 == test_iter:
                    break
            print('\n')


            # display test result
            loss_g = crit_G.smooth_loss(clear = True)
            loss_d = (crit_D_R.smooth_loss(clear = True) + crit_D_F.smooth_loss(clear = True)) / 2
            D_real = meas_D_R.smooth_loss(clear = True)
            D_fake = meas_D_F.smooth_loss(clear = True)
            D_acc = meas_acc.smooth_loss(clear = True)

            age_mae = meas_age_mae.smooth_loss(clear = True)
            ad_real = meas_age_diff_real.smooth_loss(clear = True)
            ad_fake = meas_age_diff_fake.smooth_loss(clear = True)
            fd_real = meas_feat_diff_real.smooth_loss(clear = True)
            fd_fake = meas_feat_diff_fake.smooth_loss(clear = True)

            log = '[%s] [%s] Test Epoch %d   Loss_G: %.6f   loss_D: %.6f' % \
                (time.ctime(), train_opts.id, epoch, loss_g, loss_d)
            log += '\n\tD_real: %.6f   D_fake: %.6f   D_acc: %.2f' % (D_real, D_fake, D_acc * 100.)
            log += '\n\t Age MAE: [GT: %.2f   Real: %.2f   Fake: %.2f]' % (age_mae, ad_real, ad_fake)
            log += '\n\t Feat Diff: [Real: %.6f   Fake: %.6f]' % (fd_real, fd_fake)


            print(log)
            print(log, file = fout)

            iteration = epoch * len(train_loader)
            info['test_history'].append({
                'iteration': iteration,
                'epoch': epoch,
                'Loss_G': loss_g,
                'loss_D': loss_d,
                'D_real': D_real,
                'D_fake': D_fake,
                'D_acc': D_acc,
                'age_mae': age_mae,
                'ad_real': ad_real,
                'ad_fake': ad_fake,
                'fd_real': fd_real,
                'fd_fake': fd_fake,
                })

            if train_opts.pavi == 1:
                pavi_outputs = {
                    'Loss_G': loss_g,
                    'loss_D': loss_d,
                    'delta': fd_fake,
                    'D_real_upper': D_real,
                    'D_fake_upper': D_fake,
                    'D_acc_upper': D_acc,
                }
                pavi.log(phase = 'test', iter_num = iteration, outputs = pavi_outputs)


        if train_opts.snapshot_interval > 0 and epoch % train_opts.snapshot_interval == 0:
            _snapshot(epoch)

    # final snapshot
    _snapshot(epoch = 'final')

    

def train_gan(model, train_opts):

    if not train_opts.id.startswith('gan_'):
        train_opts.id = 'gan_' + train_opts.id

    opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()

    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = 2)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        max_len = 2)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = 16, 
        num_workers = 4, pin_memory = True)

    ### create optimizer
    # use Adam
    optimizer_G = torch.optim.Adam(model.G_net.parameters(), lr = train_opts.lr * train_opts.G_lr_mult, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
            eps = train_opts.optim_epsilon)

    optimizer_D = torch.optim.Adam(model.D_net.parameters(), lr = train_opts.lr * train_opts.D_lr_mult, betas = (train_opts.optim_alpha, train_opts.optim_beta),
            eps = train_opts.optim_epsilon)

    ### loss function
    crit_G = misc.Smooth_Loss(nn.BCELoss()) # G Loss
    crit_L2 = misc.Smooth_Loss(misc.BlankLoss()) # feat_res L2_norm
    crit_D_R = misc.Smooth_Loss(nn.BCELoss()) # D loss with feat_real
    crit_D_F = misc.Smooth_Loss(nn.BCELoss()) # D loss with feat_fake

    # GAN observer
    meas_D_R = misc.Smooth_Loss(misc.BlankLoss()) # D(feat_real)
    meas_D_F1 = misc.Smooth_Loss(misc.BlankLoss()) # D(feat_fake) when training D
    meas_D_F2 = misc.Smooth_Loss(misc.BlankLoss()) # D(feat_fake) when training G
    meas_acc = misc.Smooth_Loss(misc.BCEAccuracy()) # D classification acc
    
    # Age observer
    meas_age_mae = misc.Smooth_Loss(nn.L1Loss()) # age estimation MAE
    meas_age_diff_real = misc.Smooth_Loss(nn.L1Loss()) # age estimation difference between feat_in and feat_real
    meas_age_diff_fake = misc.Smooth_Loss(nn.L1Loss()) # age estimation difference between feat_in and feat_fake

    # Feature norm observer
    meas_feat_diff_fake = misc.Smooth_Loss(misc.BlankLoss()) # (feat_res norm) / (feat_in norm)
    meas_feat_diff_real = misc.Smooth_Loss(misc.BlankLoss()) # (feat_real-feat_in norm) / (feat_in norm)
    meas_grad_norm = misc.Smooth_Loss(misc.BlankLoss()) # (feat_res grad norm)

    ### output information
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
        fn_snapshot = os.path.join(output_dir, '%s.pth' % epoch)
        print('saving checkpoint to %s' % fn_snapshot)
        model.save_model(fn_snapshot)
        io.save_json(info, fn_info)


    # text_log
    fout = open(os.path.join(output_dir, 'log.txt'), 'w')
    print(opts_str, file = fout)

    # pavi_log
    if train_opts.pavi == 1:
        pavi = PaviClient(username = 'ly015', password = '123456')
        pavi.connect(model_name = train_opts.id, info = {'session_text': opts_str})


    ### main training loop
    real_label = 1.
    fake_label = 0.

    epoch = 0
    model.eval()
    while epoch < train_opts.max_epochs:

        # set model mode
        model.G_net.train()
        model.D_net.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        optimizer_G.param_groups[0]['lr'] = lr * train_opts.G_lr_mult
        optimizer_D.param_groups[0]['lr'] = lr * train_opts.D_lr_mult
        
        # train one epoch
        for batch_idx, data in enumerate(train_loader):
            iteration = batch_idx + epoch*len(train_loader)

            ### extract feature
            img_pair, seq_len, age_gt, _ = data
            img_pair = Variable(img_pair).cuda()
            age_gt = Variable(age_gt.float()).cuda()

            bsz = img_pair.size(0) * 2

            age_out, _, feat = model.forward_video(img_pair, seq_len)
            feat.detach_()
            age_out.detach_()

            feat_in = torch.cat((feat[:,0,:], feat[:,1,:]))
            feat_real = torch.cat((feat[:,1,:], feat[:,0,:]))
            age_in = torch.cat((age_out[:,0], age_out[:,1]))
            age_real = torch.cat((age_out[:,1], age_out[:,0]))
            age_gt = torch.cat((age_gt, age_gt))

            #### train D_net
            optimizer_D.zero_grad()

            # train with real
            if model.opts.D_mode == 'cond':
                    out = model.D_net(torch.cat((feat_in, feat_real), dim = 1))
            else:
                out = model.D_net(feat_real)

            label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()
            
            loss_real = crit_D_R(out, label)
            _ = meas_acc(out, label)
            _ = meas_D_R(out, None)
            

            loss_real.backward()

            # train with fake
            noise = Variable(torch.FloatTensor(bsz, model.opts.noise_dim).normal_(0, 1)).cuda()
            feat_res = model.G_net(torch.cat((feat_in, noise), dim = 1))
            feat_fake = F.relu(feat_in + feat_res)

            if model.opts.D_mode == 'cond':
                out = model.D_net(torch.cat((feat_in, feat_fake.detach()), dim = 1))
            else:
                out = model.D_net(feat_fake.detach())

            label = Variable(torch.FloatTensor(bsz, 1).fill_(fake_label)).cuda()

            loss_fake = crit_D_F(out, label)

            _ = meas_acc(out, label)
            _ = meas_D_F1(out, None)

            loss_fake.backward()

            # update D_net
            optimizer_D.step()

            
            ### train generator
            optimizer_G.zero_grad()
            feat_fake.retain_grad() # grad of feat_fake is quivalent to grad of feat_res
            if model.opts.D_mode == 'cond':
                out = model.D_net(torch.cat((feat_in, feat_fake), dim = 1))
            else:
                out = model.D_net(feat_fake)
            label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()
            
            # G loss
            loss_g = crit_G(out, label)

            # L2 norm
            l2_res = (feat_fake - feat_in).norm(p = 2, dim = 1, keepdim = True)
            l2_diff = (feat_real - feat_in).norm(p = 2, dim = 1, keepdim = True)
            l2_threshold = Variable(torch.FloatTensor(l2_res.size()).fill_(l2_diff.max().data[0])).cuda()

            loss_l2 = crit_L2(F.relu(l2_res - l2_threshold), None) * train_opts.G_l2_weight
            

            _ = meas_D_F2(out, None)     

            loss_g = loss_g + loss_l2
            loss_g.backward()
            optimizer_G.step()

            ### observations
            age_fake, _ = model._forward_age_cls(feat_fake)
            _ = meas_age_mae(age_in, age_gt)
            _ = meas_age_diff_real(age_real, age_in)
            _ = meas_age_diff_fake(age_fake, age_in)


            feat_norm = feat_in.norm(p = 2, dim = 1, keepdim = True)
            _ = meas_feat_diff_fake(l2_res, None)
            _ = meas_feat_diff_real(l2_diff, None)
            _ = meas_grad_norm(feat_fake.grad.norm(p = 2, dim = 1, keepdim = True), None)


            ### display
            if batch_idx % train_opts.display_interval == 0:

                loss_g = crit_G.smooth_loss(clear = True)
                loss_d = (crit_D_R.smooth_loss(clear = True) + crit_D_F.smooth_loss(clear = True)) / 2
                loss_l2 = crit_L2.smooth_loss(clear = True)
                D_real = meas_D_R.smooth_loss(clear = True)
                D_fake_1 = meas_D_F1.smooth_loss(clear = True)
                D_fake_2 = meas_D_F2.smooth_loss(clear = True)
                D_acc = meas_acc.smooth_loss(clear = True)

                age_mae = meas_age_mae.smooth_loss(clear = True)
                ad_real = meas_age_diff_real.smooth_loss(clear = True)
                ad_fake = meas_age_diff_fake.smooth_loss(clear = True)
                fd_real = meas_feat_diff_real.smooth_loss(clear = True)
                fd_fake = meas_feat_diff_fake.smooth_loss(clear = True)
                grad_g = meas_grad_norm.smooth_loss(clear = True)


                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)] LR: %.3e   Loss_G: %.6f   loss_D: %.6f   loss_L2: %.6f' % \
                    (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset), 100.*batch_idx / len(train_loader),
                        lr, loss_g, loss_d, loss_l2)
                log += '\n\tD_real: %.6f   D_fake: %.6f / %.6f   D_acc: %.2f' % (D_real, D_fake_1, D_fake_2, D_acc * 100.)
                log += '\n\t Age MAE: [GT: %.2f   Real: %.2f   Fake: %.2f]' % (age_mae, ad_real, ad_fake)
                log += '\n\t Feat Diff: [Real: %.6f   Fake: %.6f]  Feat Grad: %.6f' % (fd_real, fd_fake, grad_g)

                print(log)
                print(log, file = fout)

                
                info['train_history'].append({
                    'iteration': iteration,
                    'epoch': epoch,
                    'Loss_G': loss_g,
                    'loss_D': loss_d,
                    'D_real': D_real,
                    'D_fake_1': D_fake_1,
                    'D_fake_2': D_fake_2,
                    'D_acc': D_acc,
                    'age_mae': age_mae,
                    'ad_real': ad_real,
                    'ad_fake': ad_fake,
                    'fd_real': fd_real,
                    'fd_fake': fd_fake,
                    'grad_g': grad_g
                    })

                if train_opts.pavi == 1:
                    pavi_outputs = {
                        'Loss_G': loss_g,
                        'loss_D': loss_d,
                        'delta': fd_fake,
                        'D_real_upper': D_real,
                        'D_fake_upper': D_fake_1,
                        'D_acc_upper': D_acc,
                    }
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)

        ### update epoch index
        epoch += 1

        ### test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test mode
            model.eval()

            # clear loss buffer
            for lb in [crit_G, crit_D_R, crit_D_F, meas_D_R, meas_D_F1, meas_D_F2, meas_acc,\
                        meas_age_mae, meas_age_diff_real, meas_age_diff_fake,\
                        meas_feat_diff_real, meas_feat_diff_fake, meas_grad_norm]:
                lb.clear()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)

            for batch_idx, data in enumerate(test_loader):


                img_pair, seq_len, age_gt, _ = data
                img_pair = Variable(img_pair).cuda()
                age_gt = Variable(age_gt.float()).cuda()

                bsz = img_pair.size(0) * 2

                age_out, _, feat = model.forward_video(img_pair, seq_len)
                feat.detach_()
                age_out.detach_()

                feat_in = torch.cat((feat[:,0,:], feat[:,1,:]))
                feat_real = torch.cat((feat[:,1,:], feat[:,0,:]))
                age_in = torch.cat((age_out[:,0], age_out[:,1]))
                age_real = torch.cat((age_out[:,1], age_out[:,0]))
                age_gt = torch.cat((age_gt, age_gt))

                noise = Variable(torch.FloatTensor(bsz, model.opts.noise_dim).normal_(0, 1)).cuda()
                feat_res = model.G_net(torch.cat((feat_in, noise), dim = 1))
                feat_fake = F.relu(feat_in + feat_res)

                # forward real
                if model.opts.D_mode == 'cond':
                    out = model.D_net(torch.cat((feat_in, feat_real), dim = 1))
                else:
                    out = model.D_net(feat_real)

                label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()

                _ = crit_D_R(out, label)
                _ = meas_acc(out, label)
                _ = meas_D_R(out, None)

                # forward fake
                if model.opts.D_mode == 'cond':
                    out = model.D_net(torch.cat((feat_in, feat_fake), dim = 1))
                else:
                    out = model.D_net(feat_fake)

                label = Variable(torch.FloatTensor(bsz, 1).fill_(fake_label)).cuda()

                _ = crit_D_F(out, label)
                _ = meas_acc(out, label)
                _ = meas_D_F1(out, None)

                label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()
                _ = crit_G(out, label)

                # observations
                age_fake, _ = model._forward_age_cls(feat_fake)
                _ = meas_age_mae(age_in, age_gt)
                _ = meas_age_diff_real(age_real, age_in)
                _ = meas_age_diff_fake(age_fake, age_in)

                feat_norm = feat_in.norm(p = 2, dim = 1, keepdim = True)
                _ = meas_feat_diff_fake((feat_fake - feat_in).norm(p = 2, dim = 1, keepdim = True), None)
                _ = meas_feat_diff_real((feat_real - feat_in).norm(p = 2, dim = 1, keepdim = True), None)


                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()

                if batch_idx + 1 == test_iter:
                    break
            print('\n')


            # display test result
            loss_g = crit_G.smooth_loss(clear = True)
            loss_d = (crit_D_R.smooth_loss(clear = True) + crit_D_F.smooth_loss(clear = True)) / 2
            D_real = meas_D_R.smooth_loss(clear = True)
            D_fake = meas_D_F1.smooth_loss(clear = True)
            D_acc = meas_acc.smooth_loss(clear = True)

            age_mae = meas_age_mae.smooth_loss(clear = True)
            ad_real = meas_age_diff_real.smooth_loss(clear = True)
            ad_fake = meas_age_diff_fake.smooth_loss(clear = True)
            fd_real = meas_feat_diff_real.smooth_loss(clear = True)
            fd_fake = meas_feat_diff_fake.smooth_loss(clear = True)

            log = '[%s] [%s] Test Epoch %d   Loss_G: %.6f   loss_D: %.6f' % \
                (time.ctime(), train_opts.id, epoch, loss_g, loss_d)
            log += '\n\tD_real: %.6f   D_fake: %.6f   D_acc: %.2f' % (D_real, D_fake_1, D_acc * 100.)
            log += '\n\t Age MAE: [GT: %.2f   Real: %.2f   Fake: %.2f]' % (age_mae, ad_real, ad_fake)
            log += '\n\t Feat Diff: [Real: %.6f   Fake: %.6f]' % (fd_real, fd_fake)


            print(log)
            print(log, file = fout)

            iteration = epoch * len(train_loader)
            info['test_history'].append({
                'iteration': iteration,
                'epoch': epoch,
                'Loss_G': loss_g,
                'loss_D': loss_d,
                'D_real': D_real,
                'D_fake': D_fake,
                'D_acc': D_acc,
                'age_mae': age_mae,
                'ad_real': ad_real,
                'ad_fake': ad_fake,
                'fd_real': fd_real,
                'fd_fake': fd_fake,
                })

            if train_opts.pavi == 1:
                pavi_outputs = {
                    'Loss_G': loss_g,
                    'loss_D': loss_d,
                    'delta': fd_fake,
                    'D_real_upper': D_real,
                    'D_fake_upper': D_fake,
                    'D_acc_upper': D_acc,
                }
                pavi.log(phase = 'test', iter_num = iteration, outputs = pavi_outputs)


        if train_opts.snapshot_interval > 0 and epoch % train_opts.snapshot_interval == 0:
            _snapshot(epoch)

    # final snapshot
    _snapshot(epoch = 'final')



def finetune_fix_cnn(model, train_opts):

    if not train_opts.id.startswith('gan_'):
        train_opts.id = 'gan_' + train_opts.id

    opts_str = opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()


    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = train_opts.video_max_len)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age])

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = 128, 
        num_workers = 4, pin_memory = True)


    ### create optimizer
    if train_opts.optim == 'sgd':
        optimizer = torch.optim.SGD(model.age_cls.parameters(), 
            lr = train_opts.lr, weight_decay = train_opts.weight_decay, momentum = train_opts.momentum)

    elif train_opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.age_cls.parameters(), 
            lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
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
    model.eval()
    
    while epoch < train_opts.max_epochs:

        # set model mode
        model.G_net.train()
        model.D_net.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # train one epoch
        for batch_idx, data in enumerate(train_loader):

            optimizer.zero_grad()

            img_seq, seq_len, age_gt, age_std = data
            img_seq = Variable(img_seq).cuda()
            seq_len = Variable(seq_len).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_std = age_std.float()

            age_label = age_gt.round().long() - model.opts.min_age
            
            
            # forward and backward
            age_out, fc_out, _ = model.forward_video(img_seq, seq_len)

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
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test model
            model.eval()

            # clear buffer
            crit_age.clear()
            meas_age.clear()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)    

            
            for batch_idx, data in enumerate(test_loader):

                img_seq, seq_len, age_gt, age_std = data
                img_seq = Variable(img_seq).cuda()
                seq_len = Variable(seq_len).cuda()
                age_gt = Variable(age_gt.float()).cuda()
                age_std = age_std.float()

                age_label = age_gt.round().long() - model.opts.min_age

                # forward
                age_out, fc_out, _ = model.forward_video(img_seq, seq_len)

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


    
    
def show_feat(model, dset = None, num_sample = 20, output_dir = None):
    '''
    visualize feature generated by model.G_net
    Args:
        model (GANModel instance)
        dset (Video_Age_Dataset)
        num_sample (int): number of samples to visualize
        output_dir (str): output path. default path is under model path
    '''
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    model.cuda()
    model.eval()
    
    ### load dataset
    if dset is None:
        dset = dataset.load_video_age_dataset(version = '2.0', subset = 'test', max_len = 2,
            crop_size = 128, age_rng = [0, 70])
        
    loader = torch.utils.data.DataLoader(dset, batch_size = 1, num_workers = 2, pin_memory = True)
    loaderiter = iter(loader)
        
    ### set output dir
    if output_dir is None:
        output_dir = os.path.joint('output', 'video_age_feat_analysis', 'gan_feat')
    
    io.mkdir_if_missing(output_dir)
    
    
    ### extract feature and output
    for idx in xrange(num_sample):
        data = loaderiter.next()
        img_seq, seq_len, age_gt, age_std = data
        img_seq = Variable(img_seq).cuda()
        seq_len = Variable(seq_len).cuda()
        
        age_out, fc_out, feat = model.forward_video(img_seq, seq_len)
        
        feat_in = feat[:,0,:]
        feat_real = feat[:,1,:]
        
        noise = Variable(torch.FloatTensor(1, model.opts.noise_dim).normal_(0,1)).cuda()
        feat_res = model.G_net(torch.cat((feat_in,noise), dim =1))
        feat_diff_real = feat_real - feat_in
        feat_diff_fake = F.relu(feat_in + feat_res) - feat_in
        
        
        ### draw
        fig = plt.figure(figsize = (10, 15))
        
        # frame t1
        ax = fig.add_subplot(5,1,1)
        img = mpimg.imread(dset.video_lst[idx]['frames'][0]['image'])
        ax.imshow(img)
        ax.set_xlabel('org: %s' % (dset.video_lst[idx]['id']))
        
        # feat_real
        ax = fig.add_subplot(5,1,2)
        ax.set_ylim([-2,2])
        ax.plot(feat_in.data.cpu().numpy().flatten())
        ax.set_xlabel('feat (l2_norm: %f)' % feat_in.norm().data[0])
        
        # feat_res
        ax = fig.add_subplot(5,1,3)
        ax.set_ylim([-2, 2])
        ax.plot(feat_res.data.cpu().numpy().flatten())
        ax.set_xlabel('feat_res_gen: %f' % feat_res.norm().data[0])
        
        
        # feat_diff_real
        ax = fig.add_subplot(5,1,4)
        ax.set_ylim([-2, 2])
        ax.plot(feat_diff_real.data.cpu().numpy().flatten())
        ax.set_xlabel('feat_res_real: %f' % feat_diff_real.norm().data[0])
        
        # feat_diff_fake
        ax = fig.add_subplot(5,1,5)
        ax.set_ylim([-2, 2])
        ax.plot(feat_diff_fake.data.cpu().numpy().flatten())
        ax.set_xlabel('feat_res_fake: %f' % feat_diff_fake.norm().data[0])
        
        output_fn = os.path.join(output_dir, 'feat_%d.jpg' % idx)
        fig.savefig(output_fn)
        print('visualizing %d / %d' % (idx, num_sample))
        
        
    

if __name__ == '__main__':

    command = opt_parser.parse_command()

    if command == 'pretrain':

        model_opts = opt_parser.parse_opts_gan_model()
        train_opts = opt_parser.parse_opts_pretrain()

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        # load pretrained model
        if not train_opts.pre_id.endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id, 'best.pth')
        else:
            fn = train_opts.pre_id

        model = GANModel(opts = model_opts)
        model.load_model(fn, modules = ['cnn'])

        pretrain(model, train_opts)

    elif command == 'pretrain_gan':
        
        model_opts = opt_parser.parse_opts_gan_model()
        train_opts = opt_parser.parse_opts_pretrain_gan()
        
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])
        if not train_opts.pre_id.endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id, 'best.pth')
        else:
            fn = train_opts.pred_id
        
        model = GANModel(opts = model_opts)
        model.load_model(fn, modules = ['cnn', 'age_cls'])

        pretrain_gan(model, train_opts)


    elif command == 'train_gan':

        model_opts = opt_parser.parse_opts_gan_model()
        train_opts = opt_parser.parse_opts_train_gan()

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        # load pretrained model
        if not train_opts.pre_id.endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id, 'best.pth')
        else:
            fn = train_opts.pre_id

        model = GANModel(opts = model_opts)
        if train_opts.gan_pretrained == 0:
            model.load_model(fn, modules = ['cnn', 'age_cls'])
        else:
            model.load_model(fn)

        train_gan(model, train_opts)
        
    
    elif command == 'finetune_fix':
        
        model_opts = opt_parser.parse_opts_gan_model()
        train_opts = opt_parser.parse_opts_finetune_fix_cnn()
        
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])
        
        # load pretrained model
        if not train_opts.pre_id.endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id, 'best.pth')
        else:
            fn = train_opts.pre_id
        
        model = GANModel(opts = model_opts)
        model.load_model(fn)
        
        finetune_fix_cnn(model, train_opts)
    
    elif command == 'retrain':
        
        model_opts = opt_parser.parse_opts_gan_model()
        retrain_opts = opt_parser.parse_opts_retrain()
        
        
        if retrain_opts.mode == 'pretrain':
            train_opts = opt_parser.parse_opts_pretrain()
        elif retrain_opts.mode == 'pretrain_gan':
            train_opts = opt_parser.parse_opts_pretrain_gan()
        elif retrain_opts.mode == 'train_gan':
            train_opts = opt_parser.parse_opts_train_gan()
        
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        # load training opts
        train_info = io.load_json(os.path.join('models', retrain_opts.id, 'info.json'))
        model_opts = opt_parser.update_opts_from_dict(model_opts, train_info['opts'])
        train_opts = opt_parser.update_opts_from_dict(train_opts, train_info['train_opts'],
            exceptions = ['gpu_id'])
        
        # load model
        model = GANModel(opts = model_opts)
        
        # load pretrained model
        if not train_opts.pre_id.endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id, 'best.pth')
        else:
            fn = train_opts.pre_id
        
        if retrain_opts.mode == 'pretrain':
            model.load_model(fn, modules = ['cnn'])
            pretrain(model, train_opts)
            
        elif retrain_opts.mode == 'pretrain_gan':
            model.load_model(fn, modules = ['cnn', 'age_cls'])
            pretrain_gan(model, train_opts)
            
        elif retrain_opts.mode == 'train_gan':
            if 'gan_pretrained' in train_opts and train_opts.gan_pretrained == 1:
                model.load_model(fn)
            else:
                model.load_model(fn, modules = ['cnn', 'age_cls'])
            train_gan(model, train_opts)
        elif retrain_opts.mode == 'finetune_fix':
            model.load_model(fn)
            finetune_fix_cnn(model, train_opts)
    
    
    elif command == 'show_feat':
        
        test_opts = opt_parser.parse_opts_test()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_opts.gpu_id])
        
        # load model
        if not test_opts.id.endswith('.pth'):
            fn = os.path.join('models', test_opts.id, 'best.pth')
            output_dir = os.path.join('models', test_opts.id, 'show_feat')
        else:
            fn = test_opts.id
            output_dir = os.path.join(os.path.dirname(fn), 'show_feat')
        
        model = GANModel(fn = fn)
        
        show_feat(model, output_dir = output_dir)
        
    else:
        raise Exception('invalid command "%s"' % command)
