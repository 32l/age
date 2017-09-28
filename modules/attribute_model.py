# Age Estimation Model

from __future__ import division, print_function

import util.io as io
import dataset
import misc
import opt_parser

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import os
import sys
import numpy as np
from collections import OrderedDict
import time


class AttributeModel(nn.Module):

    '''
    basic age model
    '''

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

        print('[AttributeModel.init] fn: %s' % fn)
        print('[AttributeModel.init] fn_cnn: %s' % fn_cnn)
        print('[AttributeModel.init] opts: %s' % opts)


        super(AttributeModel, self).__init__()


        ## set model opts

        if fn:
            opts = torch.load(fn, map_location=lambda storage, loc: storage)['opts']
        
        self.opts = opts

        ## load attribute lst
        self.attr_name_lst = io.load_str_list(opts.attr_name_fn)
        assert len(self.attr_name_lst) == opts.num_attr


        ## create model
        # cnn
        if opts.cnn == 'resnet18':
            net = torchvision.models.resnet18(pretrained = True)
            cnn_layers = net._modules
            cnn_layers.popitem() # remove last fc layer
            self.cnn = nn.Sequential(cnn_layers)
            self.feat_size = 512

        elif opts.cnn == 'resnet50':
            net = torchvision.models.resnet50(pretrained = True)
            cnn_layers = net._modules
            cnn_layers.popitem() # remove last fc layer
            self.cnn = nn.Sequential(cnn_layers)
            self.feat_size = 2048

        elif opts.cnn == 'vgg16':
            net = torchvision.model.vgg16(pretrained = True)
            cnn_layers = net.features._modules
            # replace the last maxpooling layer (kernel_sz = 2, stride = 2) with a more spares one.
            cnn_layers['30'] = nn.MaxPool2d(kernel_size = (4, 4), stride = (4, 4), padding = (1, 1))
            self.cnn = nn.Sequential(cnn_layers)
            self.feat_size = 8192 #(512 * 4 * 4)

        else:
            raise Exception('invalid cnn type %s' % opts.cnn)

        # feature embedding
        input_size = self.feat_size
        fc_layers = OrderedDict()
        if opts.num_fc > 0:
            assert len(opts.fc_sizes) == opts.num_fc, 'opts.fc_sizes dose not match opts.num_fc'
            for n in range(0, opts.num_fc):
                fc_layers['fc%d' % n] = nn.Linear(input_size, opts.fc_sizes[n], bias = True)
                fc_layers['fc%d_relu' % n] = nn.ReLU()
                fc_layers['fc%d_drop' % n] = nn.Dropout(p = opts.dropout)
                input_size = opts.fc_sizes[n]

            self.feat_embed = nn.Sequential(fc_layers)

        else:
            self.feat_embed = None

        # classifier
        self.attr_cls = nn.Linear(input_size, opts.num_attr, bias = True)


        # init weights
        if fn:
            print('[AttributeModel.init] loading weights from %s' % fn)
            model_info = torch.load(fn, map_location=lambda storage, loc: storage)
            self.cnn.load_state_dict(model_info['cnn_state_dict'])
            self.feat_embed.load_state_dict(model_info['feat_embed_state_dict'])
            self.attr_cls.load_state_dict(model_info['attr_cls_state_dict'])
        elif fn_cnn:
            print('[AttributeModel.init] loading CNN weights from %s' % fn_cnn)
            model_info = torch.load(fn_cnn, map_location=lambda storage, loc: storage)
            self.cnn.load_state_dict(model_info['cnn_state_dict'])

            if self.feat_embed is not None:
                if model_info['feat_embed_state_dict'] is not None:
                    self.feat_embed.load_state_dict(model_info['feat_embed_state_dict'])
                else:
                    self._init_weight(self.feat_embed)
            self._init_weight(self.attr_cls)
        else:
            self._init_weight(self.feat_embed)
            self._init_weight(self.attr_cls)


    def _init_weight(self, model = None):

        if model is None:
            model = self

        for layer in model.modules():
            for p_name, p in layer._parameters.iteritems():
                if p is not None:
                    if p_name == 'weight':
                        nn.init.xavier_normal(p.data)
                        # nn.init.normal(p.data, 0, 0.001)
                        # nn.init.uniform(p.data, -0.08, 0.08)
                        # pass
                    elif p_name == 'bias':
                        nn.init.constant(p.data, 0)

    
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
            'feat_embed_state_dict': None if self.feat_embed is None else self._get_state_dict(self.feat_embed),
            'attr_cls_state_dict': self._get_state_dict(self.attr_cls)
        }

        torch.save(model_info, fn)

    def forward(self, img):
        '''
        Forward process.
        
        Input:
            img: (bsz, 3, 224, 224). Image data mini-batch
        Output:
            attr: (bsz, num_attr). Predicted age.
        '''

        cnn_feat = self.cnn(img)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        if self.feat_embed is None:
            feat = cnn_feat
        else:
            feat = self.feat_embed(cnn_feat)
        
        attr = self.attr_cls(feat)
        attr = F.sigmoid(attr)

        return attr

    def forward_video(self, img_seq, seq_len):
        '''
        Forward video clips.
        
        Input:
            img_seq: (bsz, max_len, 3, 224, 224). Video data mini-batch
            seq_len: (bsz,). Length of each video clip.
        Output:
            attr_out: (bsz, max_len, num_attr). Predicted attribute.
        '''

        bsz, max_len = img_seq.size()[0:2]

        img_seq = img_seq.view(bsz * max_len, img_seq.size(2), img_seq.size(3), img_seq.size(4))
        
        attr = self.forward(img_seq)
        attr = attr.view(bsz, max_len, -1)

        return attr



def train_model(model, train_opts):

    print('[AttributeModel.train] training options: %s' % train_opts)

    if not train_opts.id.startswith('attr_'):
        train_opts.id = 'attr_' + train_opts.id

    # move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)

    model.cuda()

    # create data loader
    train_dset = dataset.load_attribute_dataset(dset_name = train_opts.attr_dataset, subset = 'train', alignment = train_opts.face_alignment,
        debug = train_opts.debug, crop_size = train_opts.crop_size)
    test_dset = dataset.load_attribute_dataset(dset_name = train_opts.attr_dataset, subset = 'test', alignment = train_opts.face_alignment,
        debug = train_opts.debug, crop_size = train_opts.crop_size)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 16, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = train_opts.batch_size, 
        num_workers = 4, pin_memory = True)


    # create optimizer
    if train_opts.optim == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.cnn.parameters()},
            {'params': model.feat_embed.parameters()},
            {'params': model.attr_cls.parameters(), 'lr': train_opts.lr * train_opts.cls_lr_multiplier}
            ], lr = train_opts.lr, weight_decay = train_opts.weight_decay, momentum = train_opts.momentum)

    elif train_opts.optim == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.cnn.parameters()},
            {'params': model.feat_embed.parameters()},
            {'params': model.attr_cls.parameters(), 'lr': train_opts.lr * train_opts.cls_lr_multiplier}
            ], lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
            eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    
    # loss function
    crit = nn.BCELoss()
    crit_ap = misc.MeanAP()
    

    # define output information
    info = {
        'opts': vars(model.opts),
        'train_opts': vars(train_opts),
        'train_history': [],
        'test_history': [],
    }

    # output dir
    output_dir = os.path.join('models', train_opts.id)
    io.mkdir_if_missing(output_dir)
    # output text log file
    fout = open(os.path.join(output_dir, 'log.txt'), 'w')
    # output json file
    fn_info = os.path.join(output_dir, 'info.json')

    def _snapshot(epoch):
        print('saving checkpoint to %s\t' % output_dir)
        model.save_model(os.path.join(output_dir, '%s.pth' % epoch))
        io.save_json(info, fn_info)

    # create loss buffer
    if train_opts.average_loss < 0:
        train_opts.average_loss = train_opts.display_interval

    loss_buffer = misc.Loss_Buffer(train_opts.average_loss)
    
    # main training loop
    epoch = 0

    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        optimizer.param_groups[0]['lr'] = lr # cnn
        optimizer.param_groups[1]['lr'] = lr # feat_embed
        optimizer.param_groups[2]['lr'] = lr * train_opts.cls_lr_multiplier # final fc
        # lr = optimizer.param_groups[0]['lr']

        # train one epoch
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            # data
            img, attr = data
            img = Variable(img).cuda()
            attr = Variable(attr.float()).cuda()
            
            # forward and backward
            output = model(img)

            loss = crit(output, attr)            
            
            loss.backward()

            # optimize
            if train_opts.clip_grad > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), train_opts.clip_grad)
                if total_grad_norm > train_opts.clip_grad:
                    print('Clip gradient: %f ==> %f' % (total_grad_norm, train_opts.clip_grad))
            
            optimizer.step()

            # display
            loss_smooth = loss_buffer(loss.data[0])

            if batch_idx % train_opts.display_interval == 0:

                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)]   LR: %.3e   Loss: %.6f' %\
                        (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size,
                        len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        lr, loss_smooth)


                print(log) # to screen
                print(log, file = fout) # to log file

                train_info = {
                    'iteration': batch_idx + epoch * len(train_loader),
                    'epoch': epoch, 
                    'loss': loss_smooth, 
                    }


                info['train_history'].append(train_info)

        # update epoch index
        epoch += 1


        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test mode
            model.eval()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)

            # test buffers
            test_loss_buffer = misc.Loss_Buffer(test_iter)
            crit_ap.clear()

            # test
            for batch_idx, data in enumerate(test_loader):

                img, attr = data
                img = Variable(img, volatile = True).cuda()
                attr = Variable(attr.float()).cuda()
                bsz = img.size(0)

                # forward
                output = model(img)
                loss = crit(output, attr)
                crit_ap.add(output, attr)
            
                ave_loss = test_loss_buffer(loss.data[0], bsz)

                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()

                if batch_idx + 1 == test_iter:
                    break

            # display

            mean_ap, ap_lst = crit_ap.compute_mean_ap()
            mean_ap_pn, ap_pn_lst = crit_ap.compute_mean_ap_pn()

            log1 = '[%s] [%s] Test Epoch %d   Loss: %.6f   MeanAP: %.2f   MeanAP_PN: %.2f' % \
                    (time.ctime(), train_opts.id, epoch, ave_loss, mean_ap, mean_ap_pn)

            log2 = 'Detials:\n' + \
                    '\n'.join(['%-20s   AP: %.2f   AP_PN: %.2f' % (attr_name, ap, ap_pn) for 
                        (attr_name, ap, ap_pn) in zip(model.attr_name_lst, ap_lst, ap_pn_lst)])

            log = '\n'.join([log1, log2])
            print('\n' + log)
            print(log, file = fout)

            test_info = {
                    'iteration': batch_idx + epoch * len(train_loader),
                    'epoch': epoch, 
                    'loss': ave_loss,
                    'mean_ap': mean_ap.tolist(),
                    'ap_lst': ap_lst.tolist(),
                    'mean_ap_pn': mean_ap_pn.tolist(),
                    'ap_pn_lst': ap_pn_lst.tolist()
                    }

            info['test_history'].append(test_info)


        # snapshot
        if train_opts.snapshot_interval > 0 and epoch % train_opts.snapshot_interval == 0:
            _snapshot(epoch)

    # final snapshot
    _snapshot(epoch = 'final')
    fout.close()

def test_model(model, test_opts):

    print('[AttributeModel.test] test options: %s' % test_opts)

    # move model to GPU and set to eval mode.
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()
    model.eval()

    # create dataloader
    test_dset = dataset.load_attribute_dataset(dset_name = 'celeba', subset = 'test', alignment = test_opts.face_alignment,
        debug = test_opts.debug, crop_size = test_opts.crop_size)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = test_opts.batch_size, num_workers = 4)

    crit = nn.BCELoss()
    crit_ap = nn.MeanAP()
    test_loss_buffer = misc.Loss_Buffer(len(test_loader))

    # define output information
    info = {
        'test_opts': vars(test_opts),
        'test_result': {}
    }

    # output dir
    if test_opts.id.endswith('.pth'):
        # test_opts.id is file name
        output_dir = os.path.dirname(test_opts.id)
    else:
        # test_opts.id is model id
        output_dir = os.path.join('models', test_opts.id)

    assert os.path.isdir(output_dir)

    fn_info = os.path.join(output_dir, 'test_info.json')


    # test
    for batch_idx, data in enumerate(test_loader):

        img, attr = data
        img = Variable(img, volatile = True).cuda()
        attr = Variable(attr.float()).cuda()
        bsz = img.size(0)

        # forward
        output = model(img)
        loss = crit(output, attr)
        crit_ap.add(output, attr)
    
        ave_loss = test_loss_buffer(loss.data[0], bsz)

        print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
        sys.stdout.flush()


    # output result
    mean_ap, ap_lst = crit_ap.compute_mean_ap()
    mean_ap_pn, ap_pn_lst = crit_ap.compute_mean_ap_pn()

    print('[AttributeModel.test] Test model id: %s   Loss: %.6f   meanAP: %.2f   MeanAP_PN: %.2f' %\
        (test_opts.id, ave_loss, mean_ap, mean_ap_pn))
    print('Detials:\n' + \
        '\n'.join(['%-20s   AP: %.2f   AP_PN: %.2f' % (attr_name, ap, ap_pn) for 
        (attr_name, ap, ap_pn) in zip(model.attr_name_lst, ap_lst, ap_pn_lst)]))

    info['test_result'] = {
        'loss': ave_loss,
        'mean_ap': mean_ap.tolist(),
        'ap_lst': ap_lst.tolist(),
        'mean_ap_pn': mean_ap_pn.tolist(),
        'ap_pn_lst': ap_pn_lst.tolist(),
        'attr_name_lst': model.attr_name_lst
    }

    io.save_json(info, fn_info)



def test_model_video(model, test_opts):
    print('[AttributeModel.test] test options: %s' % test_opts)

    # move model to GPU and set to eval mode.
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()
    model.eval()

    # create dataloader
    test_dset = dataset.load_video_age_dataset(version = test_opts.dataset_version, subset = test_opts.subset, 
        crop_size = test_opts.crop_size, age_rng = [0, 70])
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size = test_opts.batch_size, num_workers = 4)

    attr_pred = []
    for batch_idx, data in enumerate(test_loader):

        img_seq, seq_len, _, _  = data
        img_seq = Variable(img_seq, volatile = True).cuda()
        seq_len = Variable(seq_len, volatile = True).cuda()

        attr = model.forward_video(img_seq, seq_len)

        for i, l in enumerate(seq_len):
            l = int(l.data[0])
            attr_pred.append(attr.data.cpu()[i, 0:l, :].numpy().tolist())
        print('\rTesting %d/%d (%.2f%%)' % (batch_idx, len(test_loader), 100.*batch_idx/len(test_loader)), end = '')
        sys.stdout.flush()
    print('\n')

    id_lst = test_dset.id_lst
    rst = {s_id:p for s_id, p in zip(id_lst, attr_pred)}

    # output result
    if test_opts.id.endswith('.pth'):
        # test_opts.id is a file name
        output_dir = os.path.dirname(test_opts.id)
    else:
        # test_opts.id is a model id
        output_dir = os.path.join('models', test_opts.id)

    assert os.path.isdir(output_dir)

    fn_rst = os.path.join(output_dir, 'video_test_rst.pkl')
    io.save_data(rst, fn_rst)


if __name__ == '__main__':

    command = opt_parser.parse_command()

    if command == 'train':
        model_opts = opt_parser.parse_opts_attribute_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        model = AttributeModel(opts = model_opts)
        train_model(model, train_opts)


    elif command == 'finetune':
        model_opts = opt_parser.parse_opts_attribute_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        assert len(train_opts.pre_id) > 0, 'train_opts.pre_id not set'

        if not train_opts.pre_id[0].endswith('.pth'):
            # convert model_id to model_file
            fn = os.path.join('models', train_opts.pre_id[0], 'final.pth')
        else:
            fn = train_opts.pre_id[0]

        if train_opts.only_load_cnn == 0:
            fn_cnn = None
        else:
            fn_cnn = fn
            fn = None

        model = AttributeModel(opts = model_opts, fn = fn, fn_cnn = fn_cnn)
        train_model(model, train_opts)


    elif command == 'test':
        test_opts = opt_parser.parse_opts_test()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_opts.gpu_id])

        if test_opts.id.endswith('.pth'):
            fn = test_opts.id
        else:
            fn = os.path.join('models', test_opts.id, 'final.pth')

        model = AttributeModel(fn = fn)
        test_model(model, test_opts)

    elif command == 'test_video':
        test_opts = opt_parser.parse_opts_test()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_opts.gpu_id])

        if test_opts.id.endswith('.pth'):
            fn = test_opts.id
        else:
            fn = os.path.join('models', test_opts.id, 'final.pth')

        model = AttributeModel(fn = fn)
        test_model_video(model, test_opts)

    else:
        raise Exception('invalid command "%s"' % command)
