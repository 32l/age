# Joint model for age estimation and attribute (pose) prediction

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


class JointModel(nn.Module):

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

        print('[JointModel.init] fn: %s' % fn)
        print('[JointModel.init] fn_cnn: %s' % fn_cnn)
        print('[JointModel.init] opts: %s' % opts)


        super(JointModel, self).__init__()


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
            self.cnn_feat_size = 512

        elif opts.cnn == 'resnet50':
            net = torchvision.models.resnet50(pretrained = True)
            cnn_layers = net._modules
            cnn_layers.popitem() # remove last fc layer
            self.cnn = nn.Sequential(cnn_layers)
            self.cnn_feat_size = 2048

        elif opts.cnn == 'vgg16':
            net = torchvision.model.vgg16(pretrained = True)
            cnn_layers = net.features._modules
            # replace the last maxpooling layer (kernel_sz = 2, stride = 2) with a more spares one.
            cnn_layers['30'] = nn.MaxPool2d(kernel_size = (4, 4), stride = (4, 4), padding = (1, 1))
            self.cnn = nn.Sequential(cnn_layers)
            self.cnn_feat_size = 8192 #(512 * 4 * 4)

        else:
            raise Exception('invalid cnn type %s' % opts.cnn)


        # feature embedding
        self.feat_size = opts.feat_size
        self.feat_embed = nn.Linear(self.cnn_feat_size, self.feat_size, bias = False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = opts.dropout)



        # age estimation
        if opts.cls_type == 'oh':
            output_size = opts.max_age - opts.min_age
        elif opts.cls_type == 'dex':
            output_size = opts.max_age - opts.min_age + 1
        elif opts.cls_type == 'reg':
            output_size = 2
        else:
            raise Exception('invalid age classifier type: %s' % opts.cls_type)

        if opts.num_cls_layer == 1:
            self.age_cls = nn.Linear(self.feat_size, output_size, bias = True)
        elif opts.num_cls_layer == 2:
            self.age_cls = nn.Sequential(OrderedDict([
                    ('embed', nn.Linear(self.feat_size, opts.cls_mid_size, bias = True)),
                    ('relu', nn.ReLU()),
                    ('dropout', nn.Dropout(p = opts.dropout)),
                    ('cls', nn.Linear(opts.cls_mid_size, output_size, bias = True))
                    ]))

        # pose regression
        if opts.pose_cls == 1:
            if opts.num_cls_layer == 1:
                self.pose_cls = nn.Linear(self.feat_size, opts.pose_dim, bias = True)
            elif opts.num_cls_layer == 2:
                self.pose_cls = nn.Sequential(OrderedDict([
                    ('embed', nn.Linear(self.feat_size, opts.cls_mid_size, bias = True)),
                    ('relu', nn.ReLU()),
                    ('dropout', nn.Dropout(p = opts.dropout)),
                    ('cls', nn.Linear(opts.cls_mid_size, opts.pose_dim, bias = True))
                    ]))
        else:
            self.pose_cls = None


        # attribute recognition
        if opts.attr_cls == 1:
            self.attr_name_lst = io.load_str_list(opts.attr_name_fn)
            if opts.num_cls_layer == 1:
                self.attr_cls = nn.Linear(self.feat_size, opts.num_attr)
            elif opts.num_cls_layer == 2:
                self.attr_cls = nn.Sequential(OrderedDict([
                    ('embed', nn.Linear(self.feat_size, opts.cls_mid_size, bias = True)),
                    ('relu', nn.ReLU()),
                    ('dropout', nn.Dropout(p = opts.dropout)),
                    ('cls', nn.Linear(opts.cls_mid_size, opts.num_attr, bias = True))
                    ]))
            
        else:
            self.attr_name_lst = None
            self.attr_cls = None

            


        # init weight
        if fn:
            print('[JointModel.init] loading weights from %s' % fn)
            self.load_model(fn)
        elif fn_cnn:
            print('[JointModel.init] loading CNN weights from %s' % fn_cnn)
            self.load_model(fn_cnn, cnn_only = True)
        else:
            print('[JointModel.init] Random initialize parameters')
            self._init_weight(self.feat_embed)
            self._init_weight(self.age_cls)
            if opts.pose_cls == 1:
                self._init_weight(self.pose_cls)
            if opts.attr_cls == 1:
                self._init_weight(self.attr_cls)


    def _init_weight(self, model):

        if model is None:
            return

        for layer in model.modules():
            for p_name, p in layer._parameters.iteritems():
                if p is not None:
                    if p_name == 'weight':
                        nn.init.xavier_normal(p.data)
                    elif p_name == 'bias':
                        nn.init.constant(p.data, 0)


    def _get_state_dict(self, model):
        
        if model is None:
            return None

        state_dict = OrderedDict()
        for p_name, p in model.state_dict().iteritems():
            # remove the prefix "module.", which is added when using DataParalell for multi-gpu training
            p_name = p_name.replace('module.', '')
            state_dict[p_name] = p.cpu()

        return state_dict


    def save_model(self, fn):

        model_info = {
            'opts': self.opts,
            'state_dict_cnn': self._get_state_dict(self.cnn),
            'state_dict_feat_embed': None if self.feat_embed is None else self._get_state_dict(self.feat_embed),
            'state_dict_age_cls': self._get_state_dict(self.age_cls),
            'state_dict_pose_cls': self._get_state_dict(self.pose_cls),
            'state_dict_attr_cls': self._get_state_dict(self.attr_cls)
        }

        torch.save(model_info, fn)


    def load_model(self, fn, cnn_only = False):

        model_info = torch.load(fn, map_location = lambda storage, loc: storage)

        # load cnn, embedding layer, pose cls and attribute cls
        self.cnn.load_state_dict(model_info['state_dict_cnn'])
        self.feat_embed.load_state_dict(model_info['state_dict_feat_embed'])
        if self.pose_cls is not None:
            self.pose_cls.load_state_dict(model_info['state_dict_pose_cls'])

        if self.attr_cls is not None:
            self.attr_cls.load_state_dict(model_info['state_dict_attr_cls'])

        # if cnn_only is True, don't load age_cls (last layer)
        if cnn_only:
            self._init_weight(self.age_cls)
            
            if self.opts.num_cls_layer == 2 and \
                model_info['opts']['num_cls_layer'] == 2 and \
                self.opts.cls_mid_size == model_info['opts']['cls_mid_size']:

                # load the first layer in age cls
                self.age_cls.embed.weights.data.copy_(model_info['state_dict_age_cls']['embed.weight'])
                self.age_cls.embed.bias.data.copy_(model_info['state_dict_age_cls']['embed.bias'])
        else:
            self.age_cls.load_state_dict(model_info['state_dict_age_cls'])


            


    def forward(self, data):
        '''
        Forward process

        Input:
            data: (img_age, img_pose, img_attr)
                img_age: (bsz_age, 3, 224, 224)
                img_pose: (bsz_pose, 3, 224, 224)
                img_attr: (bsz_attr, 3, 224, 224)
        Output:
            age_out
            age_fc_out
            pose_out
            attr_out
        '''

        # data
        img = []

        if data[0] is None:
            bsz_age = 0
        else:
            bsz_age = data[0].size(0)
            img.append(data[0])

        if data[1] is None:
            bsz_pose = 0
        else:
            bsz_pose = data[1].size(0)
            img.append(data[1])

        if data[2] is None:
            bsz_attr = 0
        else:
            bsz_attr = data[2].size(0)
            img.append(data[2])

        assert len(img) > 0, 'no input image data'
        img = torch.cat(img)


        # forward
        cnn_feat = self.cnn(img)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)

        feat = self.feat_embed(cnn_feat)
        feat_relu = self.dropout(self.relu(feat))

        
        # age
        if bsz_age == 0:
            age_out = None
            age_fc_out = None
        else:
            age_fc_out = self.age_cls(feat_relu[0:bsz_age])
            if self.opts.cls_type == 'dex':
                # deep expectation
                
                age_scale = np.arange(self.opts.min_age, self.opts.max_age + 1, 1.0)
                age_scale = Variable(age_fc_out.data.new(age_scale)).unsqueeze(1)
                
                age_out = torch.matmul(F.softmax(age_fc_out), age_scale).view(-1)
                
            elif self.opts.cls_type == 'oh':
                # ordinal hyperplane
                age_fc_out = F.sigmoid(age_fc_out)
                age_out = age_fc_out.sum(dim = 1) + self.opts.min_age

            elif self.opts.cls_type == 'reg':

                age_out = self.age_fc_out.view(-1) + self.opts.min_age


        # pose
        if bsz_pose == 0 or self.pose_cls is None:
            pose_out = None
        else:
            pose_out = self.pose_cls(feat_relu[bsz_age:(bsz_age+bsz_pose)])


        # attribute
        if bsz_attr == 0 or self.attr_cls is None:
            attr_out = None
        else:
            attr_out = F.sigmoid(self.attr_cls(feat_relu[(bsz_age+bsz_pose)::]))


        return age_out, age_fc_out, pose_out, attr_out



def train_model(model, train_opts):

    print('[JointModel.train] training options: %s' % train_opts)
    if not train_opts.id.startswith('joint_'):
        train_opts.id = 'joint_' + train_opts.id

    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()

    ### create data loader
    # age
    age_train_dset = dataset.load_age_dataset(dset_name = train_opts.dataset, subset = 'train', debug = train_opts.debug,
        alignment = train_opts.face_alignment, age_rng = [model.opts.min_age, model.opts.max_age], crop_size = train_opts.crop_size)
    age_test_dset  = dataset.load_age_dataset(dset_name = train_opts.dataset, subset = 'test', debug = train_opts.debug,
        alignment = train_opts.face_alignment, age_rng = [model.opts.min_age, model.opts.max_age], crop_size = train_opts.crop_size)

    age_train_loader = torch.utils.data.DataLoader(age_train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    age_test_loader  = torch.utils.data.DataLoader(age_test_dset, batch_size = train_opts.batch_size, 
        num_workers = 4, pin_memory = True)

    # special dataset
    if train_opts.dataset in {'lap'}:
        use_age_std = True
    else:
        use_age_std = False

    # pose
    if train_opts.train_pose == 1:
        pose_train_dset = dataset.load_pose_dataset(dset_name = 'aflw', subset = 'train', alignment = train_opts.face_alignment, 
            debug = train_opts.debug, crop_size = train_opts.crop_size)
        pose_test_dset = dataset.load_pose_dataset(dset_name = 'aflw', subset = 'test', alignment = train_opts.face_alignment,
            debug = train_opts.debug, crop_size = train_opts.crop_size)

        pose_train_loader = torch.utils.data.DataLoader(pose_train_dset, batch_size = train_opts.batch_size, shuffle = True, 
            num_workers = 4, pin_memory = True, drop_last = True)
        pose_test_loader  = torch.utils.data.DataLoader(pose_test_dset, batch_size = train_opts.batch_size, 
            num_workers = 4, pin_memory = True)

        pose_train_loaderiter = iter(pose_train_loader)


    else:
        pose_train_loader = None
        pose_test_loader = None

    # attribute
    if train_opts.train_attr == 1:
        attr_train_dset = dataset.load_attribute_dataset(dset_name = 'celeba', subset = 'train', alignment = train_opts.face_alignment, 
            debug = train_opts.debug, crop_size = train_opts.crop_size)
        attr_test_dset = dataset.load_attribute_dataset(dset_name = 'celeba', subset = 'test', alignment = train_opts.face_alignment,
            debug = train_opts.debug, crop_size = train_opts.crop_size)

        attr_train_loader = torch.utils.data.DataLoader(attr_train_dset, batch_size = train_opts.batch_size, shuffle = True, 
            num_workers = 4, pin_memory = True, drop_last = True)
        attr_test_loader  = torch.utils.data.DataLoader(attr_test_dset, batch_size = train_opts.batch_size, 
            num_workers = 4, pin_memory = True)

        attr_train_loaderiter = iter(attr_train_loader)
    else:
        attr_train_loader = None
        attr_test_loader = None



    ### create optimizer
    # learnable parameters
    learnable_params = [{'params': model.age_cls.parameters(),'lr_mult': train_opts.age_cls_multiplier}]

    if train_opts.train_cnn == 1:
        learnable_params.append({'params': model.cnn.parameters(), 'lr_mult': 1.0})

    if train_opts.train_embed == 1:
        learnable_params.append({'params': model.feat_embed.parameters(), 'lr_mult': train_opts.cls_lr_multiplier})

    if train_opts.train_pose == 1:
        assert model.pose_cls is not None
        learnable_params.append({'params': model.pose_cls.parameters(), 'lr_mult': train_opts.cls_lr_multiplier})

    if train_opts.train_attr == 1:
        assert model.attr_cls is not None
        learnable_params.append({'params': model.attr_cls.parameters(), 'lr_mult': train_opts.cls_lr_multiplier})

    # create optimizer
    if train_opts.optim == 'sgd':
        optimizer = torch.optim.SGD(learnable_params, lr = train_opts.lr, 
            weight_decay = train_opts.weight_decay, momentum = train_opts.momentum)
    elif train_opts.optim == 'adam':
        optimizer = torch.optim.Adam(learnable_params, lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
            eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)


    ### loss functions
    if model.opts.cls_type == 'dex':
        crit_age = nn.CrossEntropyLoss(ignore_index = -1)
    elif model.opts.cls_type == 'oh':
        crit_age = misc.Ordinal_Hyperplane_Loss(relaxation = model.opts.oh_relaxation, ignore_index = -1)
    elif model.opts.cls_type == 'reg':
        crit_age = nn.MSELoss()

    crit_age = misc.Smooth_Loss(crit_age)
    meas_age = misc.Cumulative_Accuracy()

    if train_opts.train_pose == 1:
        crit_pose = misc.Smooth_Loss(nn.MSELoss())
        meas_pose = misc.Pose_MAE(pose_dim = model.opts.pose_dim)

    if train_opts.train_attr == 1:
        crit_attr = misc.Smooth_Loss(nn.BCELoss())
        meas_attr = misc.MeanAP()


    ### output training information
    output_dir = os.path.join('models', train_opts.id)
    io.mkdir_if_missing(output_dir)
    fn_info = os.path.join(output_dir, 'info.json')
    fn_log = os.path.join(output_dir, 'log.txt')

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


    ### main training loop
    epoch = 0 # this is the epoch idex of age data
    fout = open(fn_log, 'w')

    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for i in xrange(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * optimizer.param_groups[i]['lr_mult']

        # train one epoch
        for batch_idx, age_data in enumerate(age_train_loader):
            optimizer.zero_grad()

            # get age data
            img_age, age_gt, (age_std, age_dist) = age_data
            img_age = Variable(img_age).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_label = age_gt.round().long() - model.opts.min_age

            # get pose data
            if train_opts.train_pose == 1:
                try:
                    img_pose, pose_gt = pose_train_loaderiter.next()
                except StopIteration:
                    pose_train_loaderiter = iter(pose_train_loader)
                    img_pose, pose_gt = pose_train_loaderiter.next()

                img_pose = Variable(img_pose).cuda()    
                pose_gt = Variable(pose_gt[:,0:model.opts.pose_dim].float()).cuda()
            else:
                img_pose = None

            # get attr data
            if train_opts.train_attr == 1:
                try:
                    img_attr, attr_gt = attr_train_loaderiter.next()
                except StopIteration:
                    attr_train_loaderiter = iter(attr_train_loader)
                    img_attr, attr_gt = attr_train_loaderiter.next()

                img_attr = Variable(img_attr).cuda()
                attr_gt = Variable(attr_gt.float()).cuda()
            else:
                img_attr = None

            data = [img_age, img_pose, img_attr]


            # forward
            age_out, age_fc_out, pose_out, attr_out = model.forward(data)

            loss = crit_age(age_fc_out, age_label) * train_opts.loss_weight_age

            if use_age_std:
                meas_age.add(age_out, age_gt, age_std)
            else:
                meas_age.add(age_out, age_gt)

            if train_opts.train_pose == 1:
                loss += crit_pose(pose_out, pose_gt) * train_opts.loss_weight_pose
                meas_pose.add(pose_out, pose_gt)

            if train_opts.train_attr == 1:
                loss += crit_attr(attr_out, attr_gt) * train_opts.loss_weight_attr
                # don't compute attribute meanAP on training set
                # meas_attr.add(attr_out, attr_gt)

            loss.backward()

            
            # optimize
            if train_opts.clip_grad > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), train_opts.clip_grad)
                if total_grad_norm > train_opts.clip_grad:
                    print('Clip gradient: %f ==> %f' % (total_grad_norm, train_opts.clip_grad))

            optimizer.step()


            # display
            if batch_idx % train_opts.display_interval == 0:

                loss_age = crit_age.smooth_loss()
                mae_age = meas_age.mae()

                crit_age.clear()
                meas_age.clear()


                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)]   LR: %.3e   GPU: %s' %\
                        (time.ctime(), train_opts.id, epoch, batch_idx * age_train_loader.batch_size,
                        len(age_train_loader.dataset), 100. * batch_idx / len(age_train_loader), lr, train_opts.gpu_id)
                
                log_age = '[Age]   Loss: %.6f (weight: %.0e)   Mae: %.2f' % (loss_age, train_opts.loss_weight_age, mae_age)
                log = log + '\n\t' + log_age
                
                train_info = {
                    'iteration': batch_idx + epoch * len(age_train_loader),
                    'epoch': epoch,
                    'loss_age': loss_age, 
                    'mae_age': mae_age
                }

                
                if train_opts.train_pose:

                    loss_pose = crit_pose.smooth_loss()
                    mae_pose = meas_pose.mae()

                    crit_pose.clear()
                    meas_pose.clear()

                    log_pose = '[Pose]   Loss: %.6f (weight: %.0e)  Details:' % (loss_pose, train_opts.loss_weight_pose)

                    for i in range(model.opts.pose_dim):
                        log_pose += '[%s: %.2f]' % (['yaw', 'pitch', 'roll'][i], mae_pose[i])

                    log = log + '\n\t' + log_pose

                    train_info['loss_pose'] = loss_pose
                    train_info['mae_pose'] = mae_pose.tolist()


                if train_opts.train_attr:
                    loss_attr = crit_attr.smooth_loss()
                    crit_attr.clear()

                    log_attr = '[Attribute]   Loss: %.6f (weight: %.0e)' % (loss_attr, train_opts.loss_weight_attr)
                    log = log + '\n\t' + log_attr

                    train_info['loss_attr'] = loss_attr

                print(log) # to screen
                print(log, file = fout) # to log file

                info['train_history'].append(train_info)

        # update epoch index
        epoch += 1

        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:
            ## test age, pose, attribute respectively

            ## set test mode
            model.eval()

            ## test age
            # clear buffer
            crit_age.clear()
            meas_age.clear()

            # test
            for batch_idx, age_data in enumerate(age_test_loader):
                img_age, age_gt, (age_std, age_dist) = age_data
                img_age = Variable(img_age).cuda()
                age_gt = Variable(age_gt.float()).cuda()
                age_label = age_gt.round().long() - model.opts.min_age

                data = [img_age, None, None]

                age_out, age_fc_out, _, _ = model.forward(data)

                loss = crit_age(age_fc_out, age_label)
                if use_age_std:
                    meas_age.add(age_out, age_gt, age_std)
                else:
                    meas_age.add(age_out, age_gt, age_std)

                print('\r Testing Age %d/%d (%.2f%%)' % (batch_idx, len(age_test_loader), 100.*batch_idx/len(age_test_loader)), end = '')
            print('\n')

            # display
            loss_age = crit_age.smooth_loss()
            mae_age = meas_age.mae()
            ca3 = meas_age.ca(3)
            ca5 = meas_age.ca(5)
            ca10 = meas_age.ca(10)
            lap_err = meas_age.lap_err()

            crit_age.clear()
            meas_age.clear()

            log = '[%s] [%s] Test Epoch %d   Loss: %.6f   Mae: %.2f  CA(3): %.2f CA(5): %.2f CA(10): %.2f LAP: %.4f' % \
                    (time.ctime(), train_opts.id, epoch, loss_age, mae_age, ca3, ca5, ca10, lap_err)

            test_info = {
                'iteration': epoch * len(age_train_loader),
                'epoch': epoch,
                'loss_age': loss_age,
                'mae_age': mae_age,
                'ca3': ca3,
                'ca5': ca5,
                'ca10': ca10
            }

            ## test pose
            if train_opts.train_pose:
                crit_pose.clear()
                meas_pose.clear()

                for batch_idx, pose_data in enumerate(pose_test_loader):
                    img_pose, pose_gt = pose_data
                    img_pose = Variable(img_pose).cuda()
                    pose_gt = Variable(pose_gt[:,0:model.opts.pose_dim].float()).cuda()
                    data = [_, img_pose, _]

                    _, _, pose_out, _ = model.forward(data)

                    loss = crit_pose(pose_out, pose_gt)
                    meas_pose.add(pose_out, pose_gt)
                    print('\r Testing Pose %d/%d (%.2f%%)' % (batch_idx, len(pose_test_loader), 100.*batch_idx/len(pose_test_loader)), end = '')
                print('\n')

                loss_pose = crit_pose.smooth_loss()
                mae_pose = meas_pose.mae()

                crit_pose.clear()
                meas_pose.clear()

                log_pose = '[Pose]   Loss: %.6f   Details:' % loss_pose
                for i in range(model.opts.pose_dim):
                        log_pose += '[%s: %.2f]' % (['yaw', 'pitch', 'roll'][i], mae_pose[i])

                log = log + '\n\t' + log_pose

                test_info['loss_pose'] = loss_pose
                test_info['mae_pose'] = mae_pose.tolist()

            ## test attribute
            if train_opts.train_attr:
                crit_attr.clear()
                meas_attr.clear()

                for batch_idx, attr_data in enumerate(attr_test_loader):
                    img_attr, attr_gt = attr_data
                    img_attr = Variable(img_attr).cuda()
                    attr_gt = Variable(attr_gt.float()).cuda()
                    data = [_, _, img_attr]

                    _, _, _, attr_out = model.forward(data)

                    loss = crit_attr(attr_out, attr_gt)
                    meas_attr.add(attr_out, attr_gt)
                    print('\r Testing Attribute %d/%d (%.2f%%)' % (batch_idx, len(attr_test_loader), 100.*batch_idx/len(attr_test_loader)), end = '')
                print('\n')

                loss_attr = crit_attr.smooth_loss()
                mean_ap_attr, ap_attr = meas_attr.compute_mean_ap()
                mean_ap_pn_attr, ap_pn_attr = meas_attr.compute_mean_ap_pn()

                crit_attr.clear()
                meas_attr.clear()

                log_attr = '[Attribute]   Loss: %.6f   MeanAP: %.2f   MeanAP_PN: %.2f' % \
                    (loss_attr, mean_ap_attr, mean_ap_pn_attr)

                log_attr_details = '\n'.join(['%-20s   AP: %.2f   AP_PN: %.2f' % (attr_name, ap, ap_pn) for 
                    (attr_name, ap, ap_pn) in zip(model.attr_name_lst, ap_attr, ap_pn_attr)])

                log = log + '\n\t' + log_attr + '\n\t' + log_attr_details

                test_info['loss_attr'] = loss_attr
                test_info['mean_ap_attr'] = mean_ap_attr.tolist()
                test_info['ap_attr'] = ap_attr.tolist()
                test_info['mean_ap_pn_attr'] = mean_ap_pn_attr.tolist()
                test_info['ap_pn_attr'] = ap_pn_attr.tolist()


            print(log)
            print(log, file = fout)

            info['test_history'].append(test_info)

        # snapshot
        if train_opts.snapshot_interval > 0 and epoch % train_opts.snapshot_interval == 0:
            _snapshot(epoch)

    # final snapshot
    _snapshot(epoch = 'final')
    fout.close()




if __name__ == '__main__':

    command = opt_parser.parse_command()

    if command == 'train':
        model_opts = opt_parser.parse_opts_joint_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        model = JointModel(opts = model_opts)
        train_model(model, train_opts)

    elif command == 'finetune':
        model_opts = opt_parser.parse_opts_joint_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        assert len(train_opts.pre_id) > 0, 'train_opts.pre_id not set'

        if not train_opts.pre_id[0].endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id[0], 'final.pth')
        else:
            fn = train_opts.pre_id[0]


