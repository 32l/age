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


class AgeModel(nn.Module):

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

        print('[AgeModel.init] fn: %s' % fn)
        print('[AgeModel.init] fn_cnn: %s' % fn_cnn)
        print('[AgeModel.init] opts: %s' % opts)


        super(AgeModel, self).__init__()


        ## set model opts

        if fn:
            opts = torch.load(fn, map_location=lambda storage, loc: storage)['opts']
        
        self.opts = opts

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

        # classifier
        if opts.num_fc > 1:
            assert len(opts.fc_sizes) + 1 == opts.num_fc, 'opts.fc_sizes dose not match opts.num_fc'

        input_size = self.feat_size

        if opts.cls_type == 'oh':
            output_size = opts.max_age - opts.min_age
        elif opts.cls_type == 'dex':
            output_size = opts.max_age - opts.min_age + 1
        elif opts.cls_type == 'reg':
            output_size = 1

        fc_layers = OrderedDict()
        for n in range(0, opts.num_fc - 1):
            fc_layers['fc%d' % n] =  nn.Linear(input_size, opts.fc_sizes[n], bias = True)
            fc_layers['fc%d_relu' % n] = nn.ReLU()
            fc_layers['fc%d_drop' % n] = nn.Dropout(p = opts.dropout)
            input_size = opts.fc_sizes[n]

        fc_layers['fc_age'] = nn.Linear(input_size, output_size, bias = True)

        self.cls = nn.Sequential(fc_layers)


        # init weights
        if fn:
            print('[AgeModel.init] loading weights from %s' % fn)
            model_info = torch.load(fn, map_location=lambda storage, loc: storage)
            self.cnn.load_state_dict(model_info['cnn_state_dict'])
            self.cls.load_state_dict(model_info['cls_state_dict'])
        else:
            if fn_cnn:
                print('[AgeModel.init] loading CNN weights from %s' % fn_cnn)
                model_info = torch.load(fn_cnn, map_location=lambda storage, loc: storage)
                self.cnn.load_state_dict(model_info['cnn_state_dict'])
            self._init_weight(self.cls)


    def _init_weight(self, model = None):

        if model is None:
            model = self

        for layer in model.modules():
            for p_name, p in layer._parameters.iteritems():
                if p is not None:
                    if p_name == 'weight':
                        # nn.init.kaiming_uniform(p.data)
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
            'cls_state_dict': self._get_state_dict(self.cls)
        }

        torch.save(model_info, fn)

    def forward(self, img):
        '''
        Forward process.
        
        Input:
            img: (bsz, 3, 224, 224). Image data mini-batch
        Output:
            age_out: (bsz, 1). Predicted age.
            fc_out: (bsz, fc_age). Output of the last fc-layer
        '''

        feat = self.cnn(img)

        feat = feat.view(feat.size(0), -1)
        fc_out = self.cls(feat)

        if self.opts.cls_type == 'dex':
            # Deep EXpectation
            fc_out = F.softmax(fc_out)

            # age_lst: (num_age, 1)
            age_lst = np.arange(self.opts.min_age, self.opts.max_age + 1, 1.0)
            age_lst = Variable(fc_out.data.new(age_lst)).unsqueeze(1)

            age_out = fc_out.matmul(age_lst).view(-1)
            fc_out = torch.log(fc_out)

        elif self.opts.cls_type == 'oh':
            # Ordinal Hyperplane
            fc_out = F.sigmoid(fc_out)
            age_out = fc_out.sum(dim = 1) + self.opts.min_age

        elif self.opts.cls_type == 'reg':
            # Regression
            age_out = fc_out.view(-1)
            age_out = age_out + self.opts.min_age
       

        return age_out, fc_out


def train_model(model, train_opts):

    print('[AgeModel.train] training options: %s' % train_opts)

    if not train_opts.id.startswith('age_'):
        train_opts.id = 'age_' + train_opts.id

    # move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()

    # create data loader
    train_dset = dataset.load_age_dataset(dset_name = train_opts.dataset, subset = 'train', debug = train_opts.debug,
        alignment = train_opts.face_alignment, age_rng = [model.opts.min_age, model.opts.max_age], crop_size = train_opts.crop_size)
    test_dset  = dataset.load_age_dataset(dset_name = train_opts.dataset, subset = 'test', debug = train_opts.debug,
        alignment = train_opts.face_alignment, age_rng = [model.opts.min_age, model.opts.max_age], crop_size = train_opts.crop_size)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = train_opts.batch_size, 
        num_workers = 4, pin_memory = True)

    # special dataset
    if train_opts.dataset in {'lap'}:
        use_age_std = True
    else:
        use_age_std = False


    # create optimizer
    if train_opts.optim == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': model.cnn.parameters()},
            {'params': model.cls.parameters(), 'lr': train_opts.lr * train_opts.cls_lr_multiplier}
            ], lr = train_opts.lr, weight_decay = train_opts.weight_decay, momentum = train_opts.momentum)

    elif train_opts.optim == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.cnn.parameters()},
            {'params': model.cls.parameters(), 'lr': train_opts.lr * train_opts.cls_lr_multiplier}
            ], lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
            eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    
    # loss function
    if model.opts.cls_type == 'dex':
        crit = nn.NLLLoss(ignore_index = -1)
    elif model.opts.cls_type == 'oh':
        crit = misc.Ordinal_Hyperplane_Loss(relaxation = model.opts.oh_relaxation, ignore_index = -1)
    elif model.opts.cls_type == 'reg':
        crit = nn.MSELoss()

    crit = misc.Smooth_Loss(crit)
    meas = misc.Cumulative_Accuracy()


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

    
    # main training loop
    epoch = 0

    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * train_opts.cls_lr_multiplier
        

        # train one epoch
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            # data
            img, age_gt, (age_std, age_dist) = data
            img = Variable(img).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_label = age_gt.round().long() - model.opts.min_age
            
            # forward and backward
            age_out, fc_out = model(img)

            loss = crit(fc_out, age_label)
            if use_age_std:
                meas.add(age_out, age_gt, age_std)
            else:
                meas.add(age_out, age_gt)

            loss.backward()

            if train_opts.clip_grad > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), train_opts.clip_grad)
                if total_grad_norm > train_opts.clip_grad:
                    print('Clip gradient: %f ==> %f' % (total_grad_norm, train_opts.clip_grad))


            # optimize
            optimizer.step()

            # display

            if batch_idx % train_opts.display_interval == 0:
                loss_smooth = crit.smooth_loss()
                mae_smooth = meas.mae()

                crit.clear()
                meas.clear()

                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)]   LR: %.3e   Loss: %.6f   Mae: %.2f' %\
                        (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size,
                        len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        lr, loss_smooth, mae_smooth)

                print(log) # to screen
                print(log, file = fout) # to log file

                batch_idx_glb = batch_idx + epoch * len(train_loader)
                info['train_history'].append({
                    'iteration': batch_idx_glb,
                    'epoch': epoch, 
                    'loss': loss_smooth, 
                    'mae': mae_smooth})

        # update epoch index
        epoch += 1


        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test model
            model.eval()

            # clean buffer
            crit.clear()
            meas.clear()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)
            

            # test
            for batch_idx, data in enumerate(test_loader):

                img, age_gt, (age_std, age_dist) = data

                img = Variable(img).cuda()
                age_gt = Variable(age_gt.float()).cuda()
                age_label = age_gt.round().long() - model.opts.min_age
                
                # forward
                age_out, fc_out = model(img)
                loss = crit(fc_out, age_label)
                
                if use_age_std:
                    meas.add(age_out, age_gt, age_std)
                else:
                    meas.add(age_out, age_gt)

                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()

                if batch_idx + 1 == test_iter:
                    break

            # display
            loss = crit.smooth_loss()
            mae = meas.mae()

            log = '[%s] [%s] Test Epoch %d   Loss: %.6f   Mae: %.2f  CA(3): %.2f CA(5): %.2f CA(10): %.2f LAP: %.4f' % \
                    (time.ctime(), train_opts.id, epoch, loss, mae, meas.ca(3), meas.ca(5), meas.ca(10), meas.lap_err())

            print('\n' + log)
            print(log, file = fout)

            info['test_history'].append({
                    'iteration': epoch * len(train_loader),
                    'epoch': epoch, 
                    'loss': loss, 
                    'mae': mae})
            
            crit.clear()
            meas.clear()

        # snapshot
        if train_opts.snapshot_interval > 0 and epoch % train_opts.snapshot_interval == 0:
            _snapshot(epoch)

    # final snapshot
    _snapshot(epoch = 'final')
    fout.close()

def test_model(model, test_opts):

    print('[AgeModel.test] test options: %s' % test_opts)

    # move model to GPU and set to eval mode.
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()
    model.eval()

    # create dataloader
    test_dset = dataset.load_age_dataset(dset_name = test_opts.dataset, subset = test_opts.subset,
        alignment = test_opts.face_alignment, age_rng = [model.opts.min_age, model.opts.max_age], crop_size = test_opts.crop_size)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = test_opts.batch_size, num_workers = 4)

    crit_acc = misc.Cumulative_Accuracy()

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
    fn_rst = os.path.join(output_dir, 'test_rst.pkl')


    # test
    age_pred = []

    for batch_idx, data in enumerate(test_loader):

        img, age_gt, (age_std, age_dist) = data
        img = Variable(img, volatile = True).cuda()
        age_gt = Variable(age_gt.float()).cuda()
        
        num_val = (age_gt != -1).data.sum()

        # forward
        age_out, fc_out = model(img)

        crit_acc.add(age_out, age_gt)

        if test_opts.output_rst == 1:
            age_pred.append(age_out.data.cpu().numpy().flatten())

        print('\rTesting %d/%d (%.2f%%)' % (batch_idx, len(test_loader), 100.*batch_idx/len(test_loader)), end = '')
        sys.stdout.flush()


    # output result
    info['test_result'] = {
        'mae': crit_acc.mae(),
        'ca3': crit_acc.ca(3),
        'ca5': crit_acc.ca(5),
        'ca10': crit_acc.ca(10)
    }

    print('[AgeModel.test] Test model id: %s' % test_opts.id)
    for metric in ['mae', 'ca3', 'ca5', 'ca10']:
        print('\t%s\t%f' % (metric, info['test_result'][metric]))

    io.save_json(info, fn_info)

    if test_opts.output_rst:
        age_pred = np.concatenate(age_pred).tolist()
        assert len(age_pred) == len(test_dset.sample_lst)

        rst_list = {s['id']:{'age': a} for s, a in zip(test_dset.sample_lst, age_pred)}
        io.save_data(rst_list, fn_rst)


if __name__ == '__main__':

    command = opt_parser.parse_command()

    if command == 'train':
        model_opts = opt_parser.parse_opts_age_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        model = AgeModel(opts = model_opts)
        train_model(model, train_opts)


    elif command == 'finetune':
        model_opts = opt_parser.parse_opts_age_model()
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

        model = AgeModel(opts = model_opts, fn = fn, fn_cnn = fn_cnn)
        train_model(model, train_opts)

    elif command == 'test':
        test_opts = opt_parser.parse_opts_test()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_opts.gpu_id])

        if test_opts.id.endswith('.pth'):
            fn = test_opts.id
        else:
            fn = os.path.join('models', test_opts.id, 'final.pth')

        model = AgeModel(fn = fn)
        test_model(model, test_opts)

    else:
        raise Exception('invalid command "%s"' % command)
