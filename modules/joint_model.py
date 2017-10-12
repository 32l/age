# Joint model for age estimation and attribute (pose) prediction

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
import opt_parser

import os
import sys
import numpy as np
from collections import OrderedDict
import time


class JointModel(nn.Module):

    def _update_opts(self, opts):
        '''
        update old version of model options
        '''

        if 'attr_share_fc' not in opts:
            opts.attr_share_fc = 0


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

        print('[JointModel.init] fn: %s' % fn)
        print('[JointModel.init] fn_cnn: %s' % fn_cnn)
        # print('[JointModel.init] opts: %s' % opts)


        super(JointModel, self).__init__()


        ## set model opts

        if fn:
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
                if opts.attr_share_fc == 1:
                    assert opts.pose_cls == 1
                    embed = self.pose_cls.embed
                else:
                    embed = nn.Linear(self.feat_size, opts.cls_mid_size, bias = True)

                self.attr_cls = nn.Sequential(OrderedDict([
                    ('embed', embed),
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
                self._init_weight(self.pose_cls, mode = 'normal')
            if opts.attr_cls == 1:
                self._init_weight(self.attr_cls)

        # set perturbation optsion
        # options:
        #     enable (bool): enable feature perturbation
        #     mode(str):
        #         'age_embed_L2'
        #         'age_embed_cos'

        self.pert_opts = {
            'enable': False,
            'mode': None, #'age_embed_L2'
            'guide_signal': None, #'pose'
            'guide_index': None, # 0
            'scale': None, # 0.1
            'debug': False,
        }

    def _init_weight(self, model = None, mode = 'xavier'):

        if model is None:
            return

        for layer in model.modules():
            for p_name, p in layer._parameters.iteritems():
                if p is not None:
                    if p_name == 'weight':
                        if mode == 'xavier':
                            nn.init.xavier_normal(p.data)
                        elif mode == 'normal':
                            nn.init.normal(p.data, 0, 0.01)
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
                model_info['opts'].num_cls_layer == 2 and \
                self.opts.cls_mid_size == model_info['opts'].cls_mid_size:

                # load the first layer in age cls
                self.age_cls.embed.weight.data.copy_(model_info['state_dict_age_cls']['embed.weight'])
                self.age_cls.embed.bias.data.copy_(model_info['state_dict_age_cls']['embed.bias'])
        else:
            self.age_cls.load_state_dict(model_info['state_dict_age_cls'])


    def _compute_age(self, feat_relu):
        '''
        input:
            feat: output of feat_embed layer (after relu)
        output:
            age_out
            age_fc_out
        '''
        age_fc_out = self.age_cls(feat_relu)

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
            # regression
            age_out = self.age_fc_out.view(-1) + self.opts.min_age

        return age_out, age_fc_out


    def perturb_feat(self, feat):
        '''
        attribute-guided feature perturbation
        input:
            feat: output of feat_embed layer (before relu)
        output:
            feat_pert: same size as feat
        '''
        opts = self.pert_opts

        if opts['guide_signal'] == 'pose':
            guide_cls = self.pose_cls
        elif opts['guide_signal'] == 'attr':
            guide_cls = self.attr_cls


        # compute guide signal
        feat = feat.detach()
        feat.requires_grad = True
        
        feat = feat.clone()
        feat.retain_grad()

        feat_relu = self.relu(feat)
        guide_out = guide_cls(feat_relu)[:, opts['guide_index']]

        # compute feature perturbation by backforward
        guide_out.sum().backward()
        feat_delta = feat.grad

        # normalization
        scale = opts['scale'] * (feat.norm(dim = 1, keepdim = True) / feat_delta.norm(dim = 1, keepdim = True))
        feat_delta_n = feat_delta * scale

        feat_delta_n.detach_()
        feat_delta_n.volatile = False
        feat_delta_n.requires_grad = False
        
        if opts['debug']:
            for s in np.arange(0.05, 0.51, 0.05).tolist():
                bsz = feat.size(0)
                scale = s * (feat.norm(dim = 1, keepdim = True) / feat_delta.norm(dim = 1, keepdim = True))
                
                feat_delta_s = feat_delta * scale

                feat_pert = self.relu(torch.cat([feat + feat_delta_s, feat - feat_delta_s]))
                guide_out_pert = guide_cls(feat_pert)[:, opts['guide_index']].contiguous()
                age_out_pert, _ = self._compute_age(feat_pert)

                # compute feat_age
                feat_age = self.age_cls.relu(self.age_cls.embed(feat))
                feat_age_pert = self.age_cls.relu(self.age_cls.embed(feat_pert))
                feat_age_delta = feat_age_pert[0:bsz] - feat_age_pert[bsz::]

                # compute feat_guide
                feat_guide = guide_cls.relu(guide_cls.embed(feat))
                feat_guide_pert = guide_cls.relu(guide_cls.embed(feat_pert))
                feat_guide_delta = feat_guide_pert[0:bsz] - feat_guide_pert[bsz::]

                mean_scale = scale.mean().data[0]
                mean_guide_pert = (guide_out_pert[0:bsz] - guide_out_pert[bsz::]).mean().data[0]
                mean_age_pert = (age_out_pert[0:bsz] - age_out_pert[bsz::]).abs().mean().data[0]


                ####### perturbation informatin #########
                print('#####################################################')
                print('scale: %f' % s)
                print('feat_perturb_scale:')
                print((feat_delta_s.norm(dim = 1) / feat.norm(dim = 1)).mean().data[0])
                
                print('feat_age_perturb_scale:')
                print((feat_age_delta.norm(dim = 1) / feat_age.norm(dim = 1)).mean().data[0])

                print('guide_perturb_scale:')
                print((feat_guide_delta.norm(dim = 1) / feat_guide.norm(dim = 1)).mean().data[0])

                print('perturbation_debug_info: mean_scale: %f,   mean_guide_pert: %f,   mean_age_pert: %f' %\
                    (mean_scale, mean_guide_pert, mean_age_pert))

            ####### within video difference #########
            print('#####################################################')
            seq_len = 8
            feat = feat.view(-1, seq_len, feat.size(1))
            feat_diff = feat[:,0:-1,:] - feat[:, 1::,:]
            

            feat_age = feat_age.view(-1, seq_len, feat_age.size(1))
            feat_age_diff = feat_age[:,0:-1,:] - feat_age[:,1::,:]

            feat_guide = feat_guide.view(-1, seq_len, feat_guide.size(1))
            feat_guide_diff = feat_guide[:,0:-1,:] - feat_guide[:,1::,:]

            age_out,_ = self._compute_age(feat_relu)
            age_out = age_out.contiguous().view(-1, seq_len)
            age_diff = age_out[:,0:-1] - age_out[:,1::]

            guide_out = guide_out.contiguous().view(-1, seq_len)
            guide_diff = guide_out[:,0:-1] - guide_out[:,1::]


            print('frame_feat_diff_scale:')
            print((feat_diff.norm(dim = 2).mean(dim = 1)/feat.norm(dim = 2).mean(dim = 1)).mean().data[0])

            print('frame_feat_age_diff_scale:')
            print((feat_age_diff.norm(dim = 2).mean(dim = 1)/feat_age.norm(dim = 2).mean(dim = 1)).mean().data[0])

            print('frame_feat_guide_diff_scale:')
            print((feat_guide_diff.norm(dim = 2).mean(dim = 1)/feat_guide.norm(dim = 2).mean(dim = 1)).mean().data[0])

            print('frame_age_diff:')
            print(age_diff.abs().mean().data[0])

            print('frame_guide_diff:')
            print(guide_diff.abs().mean().data[0])


            ###### between video difference #########
            print('-----------------------------------------------------')
            feat = feat.mean(dim = 1)
            feat_diff = feat[0:-1,:] - feat[1::,:]

            feat_age = feat_age.mean(dim = 1)
            feat_age_diff = feat_age[0:-1,:] - feat_age[1::,:]

            feat_guide = feat_guide.mean(dim = 1)
            feat_guide_diff = feat_guide[0:-1,:] - feat_guide[1::,:]

            print('video_feat_diff_scale:')
            print((feat_diff.norm(dim = 1).mean()/feat.norm(dim = 1).mean()).data[0])

            print('video_feat_age_diff_scale:')
            print((feat_age_diff.norm(dim = 1).mean()/feat_age.norm(dim = 1).mean()).data[0])

            print('video_feat_guide_diff_scale:')
            print((feat_guide_diff.norm(dim = 1).mean()/feat_guide.norm(dim = 1).mean()).data[0])
            print('#####################################################')
        
            exit(0)

        guide_cls.zero_grad()

        return feat_delta_n


    def forward(self, data, joint_test = False):
        '''
        Forward process

        Input:
            data: (img_age, img_pose, img_attr)
                img_age: (bsz_age, 3, 224, 224)
                img_pose: (bsz_pose, 3, 224, 224)
                img_attr: (bsz_attr, 3, 224, 224)

            joint_test:
                age, pose and attribute branches will use the same data (img_age)
        Output:
            age_out
            age_fc_out
            pose_out
            attr_out

            pert_out
        '''

        # data
        img = []

        if data[0] is None:
            bsz_age = 0
        else:
            bsz_age = data[0].size(0)
            img.append(data[0])


        if joint_test:
            assert bsz_age > 0
            bsz_pose = bsz_attr = bsz_age
        else:
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


        # forward cnn and common embedding
        cnn_feat = self.cnn(img)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        feat = self.feat_embed(cnn_feat)
        
        # perturbation
        if self.pert_opts['enable'] and bsz_age > 0:
            feat_comm = feat[0:bsz_age]
            feat_delta = self.perturb_feat(feat_comm)
            assert feat_comm.is_same_size(feat_delta)

            feat_comm_pert = torch.cat([feat_comm + feat_delta, feat_comm - feat_delta])
            feat_age_pert = self.age_cls.embed(self.relu(feat_comm_pert))
            pert_out = [feat_age_pert[0:bsz_age], feat_age_pert[bsz_age::]]
        else:
            pert_out = None
        

        # forward classifiers
        feat_relu = self.dropout(self.relu(feat))
            # age
        if bsz_age == 0:
            age_out = None
            age_fc_out = None
        else:
            age_out, age_fc_out = self._compute_age(feat_relu[0:bsz_age])

            # pose
        if bsz_pose == 0 or self.pose_cls is None:
            pose_out = None
        else:
            if joint_test:
                pose_out = self.pose_cls(feat_relu[0:bsz_age])
            else:
                pose_out = self.pose_cls(feat_relu[bsz_age:(bsz_age+bsz_pose)])


            # attribute
        if bsz_attr == 0 or self.attr_cls is None:
            attr_out = None
        else:
            if joint_test:
                attr_out = F.sigmoid(self.attr_cls(feat_relu[0:bsz_age]))
            else:
                attr_out = F.sigmoid(self.attr_cls(feat_relu[(bsz_age+bsz_pose)::]))


        return age_out, age_fc_out, pose_out, attr_out, pert_out


    def forward_video(self, data):
        '''
        Forward process. Data for age estimation task is in video format

        Input:
            data: ((img_seq_age, seq_len_age), img_pose, img_attr)
                img_seq_age: (bsz_age, max_len, 3, 224, 224)
                seq_len_age: (bsz_age, 1)
                img_pose: (bsz_pose, 3, 224, 224)
                img_attr: (bsz_attr, 3, 224, 224)
        Output:
            age_out: (bsz_age, max_len)
            age_fc_out: (bsz_age, max_len, *fc_sz)
            pose_out: (bsz_pose, pose_dim)
            attr_out: (bsz_attr, num_attr)
            pert_out: TBD
        '''

        # data
        if data[0] is not None:
            # unfold video frames
            img_seq_age, seq_len_age = data[0]
            bsz_age, max_len = img_seq_age.size()[0:2]
            img_sz = img_seq_age.size()[2::]

            img_seq_age = img_seq_age.view(bsz_age*max_len, *img_sz)
            data[0] = img_seq_age


        age_out, age_fc_out, pose_out, attr_out, pert_out = self.forward(data)

        if data[0] is not None:
            age_out = age_out.view(bsz_age, max_len)
            age_fc_out = age_fc_out.view(bsz_age, max_len, -1)
            
            if pert_out is not None:
                pert_out[0] = pert_out[0].view(bsz_age, max_len, -1)
                pert_out[1] = pert_out[1].view(bsz_age, max_len, -1)


        return age_out, age_fc_out, pose_out, attr_out, pert_out

    def forward_video_test(self, data):
        '''
        Forward video data through age, pose and attribute branches

        Input:
            data: (img_seq, seq_len)
                img_seq: (bsz, max_len, 3, 224, 224)
                seq_len: (bsz, 1)
        Output:
            age_out: (bsz, max_len)
            age_fc_out: (bsz, max_len, *fc_sz)
            pose_out: (bsz, pose_dim)
            attr_out: (bsz, num_attr)
            pert_out: TBD
        '''

        img_seq, seq_len = data

        bsz, max_len = img_seq.size()[0:2]
        img_sz = img_seq.size()[2::]

        img_seq = img_seq.view(bsz * max_len, *img_sz)
        data = [img_seq, None, None]

        age_out, age_fc_out, pose_out, attr_out, _ = self.forward(data, joint_test = True)

        age_out = age_out.view(bsz, max_len)
        age_fc_out = age_fc_out.view(bsz, max_len, -1)

        if pose_out is not None:
            pose_out = pose_out.view(bsz, max_len, -1)
        if attr_out is not None:
            attr_out = attr_out.view(bsz, max_len, -1)

        return age_out, age_fc_out, pose_out, attr_out


def train_model(model, train_opts):

    if not train_opts.id.startswith('joint_'):
        train_opts.id = 'joint_' + train_opts.id

    opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

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
        attr_train_dset = dataset.load_attribute_dataset(dset_name = train_opts.attr_dataset, subset = 'train', alignment = train_opts.face_alignment, 
            debug = train_opts.debug, crop_size = train_opts.crop_size)
        attr_test_dset = dataset.load_attribute_dataset(dset_name = train_opts.attr_dataset, subset = 'test', alignment = train_opts.face_alignment,
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
    learnable_params = [{'params': model.age_cls.parameters(),'lr_mult': train_opts.cls_lr_multiplier}]

    if train_opts.train_cnn == 1:
        learnable_params.append({'params': model.cnn.parameters(), 'lr_mult': 1.0})

    if train_opts.train_embed == 1:
        learnable_params.append({'params': model.feat_embed.parameters(), 'lr_mult': train_opts.sidetask_lr_multiplier})

    if train_opts.train_pose == 1:
        assert model.pose_cls is not None
        wd = train_opts.weight_decay * train_opts.loss_weight_pose if train_opts.adjust_weight_decay == 1 else train_opts.weight_decay
        p_pose = {'params': model.pose_cls.parameters(), 'lr_mult': train_opts.sidetask_lr_multiplier, 'weight_decay': wd}
        learnable_params.append(p_pose)
        
    if train_opts.train_attr == 1:
        assert model.attr_cls is not None
        if model.opts.attr_share_fc == 1:
            params = model.attr_cls.cls.parameters()
        else:
            params = model.attr_cls.parameters()
        wd = train_opts.weight_decay * train_opts.loss_weight_attr if train_opts.adjust_weight_decay == 1 else train_opts.weight_decay
        learnable_params.append({'params': params, 'lr_mult': train_opts.sidetask_lr_multiplier, 'weight_decay': wd})

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
        fn_snapshot = os.path.join(output_dir, '%s.pth' % epoch)
        print('saving checkpoint to %s' % fn_snapshot)
        model.save_model(fn_snapshot)
        io.save_json(info, fn_info)


    # text_log
    fn_log = os.path.join(output_dir, 'log.txt')
    fout = open(fn_log, 'w')
    print(opts_str, file = fout)

    # pavi_log
    if train_opts.pavi == 1:
        pavi = PaviClient(username = 'ly015', password = '123456')
        pavi.connect(model_name = train_opts.id, info = {'session_text': opts_str})



    # save checkpoint if getting a best performance
    checkbest_name = 'mae_age'
    checkbest_value = sys.float_info.max
    checkbest_epoch = -1


    ### main training loop
    epoch = 0 # this is the epoch idex of age data
    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for i in xrange(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * optimizer.param_groups[i]['lr_mult']

        if train_opts.lr_decay_pose > 0 :
            p_pose['lr'] = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay_pose)) * p_pose['lr_mult']
        

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
            age_out, age_fc_out, pose_out, attr_out, _ = model.forward(data)

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
                iteration = batch_idx + epoch * len(age_train_loader)

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
                    'iteration': iteration,
                    'epoch': epoch,
                    'loss_age': loss_age, 
                    'mae_age': mae_age
                }

                pavi_outputs = {
                    'loss_age': loss_age,
                    'mae_age_upper': mae_age,
                }

                
                if train_opts.train_pose:

                    loss_pose = crit_pose.smooth_loss()
                    mae_pose = meas_pose.mae().tolist()

                    crit_pose.clear()
                    meas_pose.clear()

                    log_pose = '[Pose]   Loss: %.6f (weight: %.0e)  Details:' % (loss_pose, train_opts.loss_weight_pose)

                    for i in range(model.opts.pose_dim):
                        log_pose += '[%s: %.2f]' % (['yaw', 'pitch', 'roll'][i], mae_pose[i])

                    log = log + '\n\t' + log_pose

                    train_info['loss_pose'] = loss_pose
                    train_info['mae_pose'] = mae_pose

                    pavi_outputs['loss_pose'] = loss_pose
                    for i in range(model.opts.pose_dim):
                        pavi_outputs['mae_pose_%s_upper' % ['yaw', 'pitch', 'roll'][i]] = mae_pose[i]


                if train_opts.train_attr:
                    loss_attr = crit_attr.smooth_loss()
                    crit_attr.clear()

                    log_attr = '[Attribute]   Loss: %.6f (weight: %.0e)' % (loss_attr, train_opts.loss_weight_attr)
                    log = log + '\n\t' + log_attr

                    train_info['loss_attr'] = loss_attr
                    pavi_outputs['loss_attr'] = loss_attr

                print(log) # to screen
                print(log, file = fout) # to log file

                info['train_history'].append(train_info)

                if train_opts.pavi == 1:                
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)


        # update epoch index
        epoch += 1

        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:
            ## test age, pose, attribute respectively
            iteration = epoch * len(age_train_loader)

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

                age_out, age_fc_out, _, _, _ = model.forward(data)

                loss = crit_age(age_fc_out, age_label)
                if use_age_std:
                    meas_age.add(age_out, age_gt, age_std)
                else:
                    meas_age.add(age_out, age_gt)

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

            log_age = '[Age]   Loss: %.6f (weight: %.0e)   Mae: %.2f  CA(3): %.2f CA(5): %.2f CA(10): %.2f LAP: %.4f' % \
                    (loss_age, train_opts.loss_weight_age, mae_age, ca3, ca5, ca10, lap_err)

            log = log + '\n\t' + log_age

            test_info = {
                'iteration': iteration,
                'epoch': epoch,
                'loss_age': loss_age,
                'mae_age': mae_age,
                'ca3': ca3,
                'ca5': ca5,
                'ca10': ca10,
                'lap_err': lap_err
            }

            pavi_outputs = {
                'loss_age': loss_age,
                'mae_age_upper': mae_age,
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

                    _, _, pose_out, _, _ = model.forward(data)

                    loss = crit_pose(pose_out, pose_gt)
                    meas_pose.add(pose_out, pose_gt)
                    print('\r Testing Pose %d/%d (%.2f%%)' % (batch_idx, len(pose_test_loader), 100.*batch_idx/len(pose_test_loader)), end = '')
                print('\n')

                loss_pose = crit_pose.smooth_loss()
                mae_pose = meas_pose.mae().tolist()

                crit_pose.clear()
                meas_pose.clear()

                log_pose = '[Pose]   Loss: %.6f   Details:' % loss_pose
                for i in range(model.opts.pose_dim):
                        log_pose += '[%s: %.2f]' % (['yaw', 'pitch', 'roll'][i], mae_pose[i])

                log = log + '\n\t' + log_pose

                test_info['loss_pose'] = loss_pose
                test_info['mae_pose'] = mae_pose

                pavi_outputs['loss_pose'] = loss_pose
                for i in range(model.opts.pose_dim):
                    pavi_outputs['mae_pose_%s_upper' % ['yaw', 'pitch', 'roll'][i]] = mae_pose[i]

            ## test attribute
            if train_opts.train_attr:
                crit_attr.clear()
                meas_attr.clear()

                for batch_idx, attr_data in enumerate(attr_test_loader):
                    img_attr, attr_gt = attr_data
                    img_attr = Variable(img_attr).cuda()
                    attr_gt = Variable(attr_gt.float()).cuda()
                    data = [_, _, img_attr]

                    _, _, _, attr_out, _ = model.forward(data)

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

                log = log + '\n\t' + log_attr + '\n\n' + log_attr_details

                test_info['loss_attr'] = loss_attr
                test_info['mean_ap_attr'] = mean_ap_attr.tolist()
                test_info['ap_attr'] = ap_attr.tolist()
                test_info['mean_ap_pn_attr'] = mean_ap_pn_attr.tolist()
                test_info['ap_pn_attr'] = ap_pn_attr.tolist()

                pavi_outputs['loss_attr'] = loss_attr
                pavi_outputs['mean_ap_attr_upper'] = mean_ap_attr.tolist()


            print(log)
            print(log, file = fout)

            info['test_history'].append(test_info)

            if train_opts.pavi == 1:
                pavi.log(phase = 'test', iter_num = iteration, outputs = pavi_outputs)

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


def train_model_video(model, train_opts):
    if not train_opts.id.startswith('joint_'):
        train_opts.id = 'joint_' + train_opts.id

    opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to GPU
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()


    ### create data loader
    # age
    age_train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = train_opts.video_max_len)
    age_test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age])

    age_train_loader = torch.utils.data.DataLoader(age_train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    age_test_loader  = torch.utils.data.DataLoader(age_test_dset, batch_size = 16, 
        num_workers = 4, pin_memory = True)

    # pose
    if train_opts.train_pose == 1:
        pose_train_dset = dataset.load_pose_dataset(dset_name = 'aflw', subset = 'train', alignment = train_opts.face_alignment, 
            debug = train_opts.debug, crop_size = train_opts.crop_size)
        pose_test_dset = dataset.load_pose_dataset(dset_name = 'aflw', subset = 'test', alignment = train_opts.face_alignment,
            debug = train_opts.debug, crop_size = train_opts.crop_size)

        pose_train_loader = torch.utils.data.DataLoader(pose_train_dset, batch_size = train_opts.batch_size_pose, shuffle = True, 
            num_workers = 4, pin_memory = True, drop_last = True)
        pose_test_loader  = torch.utils.data.DataLoader(pose_test_dset, batch_size = train_opts.batch_size_pose, 
            num_workers = 4, pin_memory = True)

        pose_train_loaderiter = iter(pose_train_loader)
    else:
        pose_train_loader = None
        pose_test_loader = None

    # attribute
    if train_opts.train_attr == 1:
        attr_train_dset = dataset.load_attribute_dataset(dset_name = train_opts.attr_dataset, subset = 'train', alignment = train_opts.face_alignment, 
            debug = train_opts.debug, crop_size = train_opts.crop_size)
        attr_test_dset = dataset.load_attribute_dataset(dset_name = train_opts.attr_dataset, subset = 'test', alignment = train_opts.face_alignment,
            debug = train_opts.debug, crop_size = train_opts.crop_size)

        attr_train_loader = torch.utils.data.DataLoader(attr_train_dset, batch_size = train_opts.batch_size_attr, shuffle = True, 
            num_workers = 4, pin_memory = True, drop_last = True)
        attr_test_loader  = torch.utils.data.DataLoader(attr_test_dset, batch_size = train_opts.batch_size_attr, 
            num_workers = 4, pin_memory = True)

        attr_train_loaderiter = iter(attr_train_loader)
    else:
        attr_train_loader = None
        attr_test_loader = None


    ### create optimizer
    # learnable parameters
    learnable_params = [{'params': model.age_cls.parameters(),'lr_mult': train_opts.cls_lr_multiplier}]

    if train_opts.train_cnn == 1:
        learnable_params.append({'params': model.cnn.parameters(), 'lr_mult': 1.0})

    if train_opts.train_embed == 1:
        learnable_params.append({'params': model.feat_embed.parameters(), 'lr_mult': train_opts.sidetask_lr_multiplier})

    if train_opts.train_pose == 1:
        assert model.pose_cls is not None
        wd = train_opts.weight_decay * train_opts.loss_weight_pose if train_opts.adjust_weight_decay == 1 else train_opts.weight_decay
        p_pose = {'params': model.pose_cls.parameters(), 'lr_mult': train_opts.sidetask_lr_multiplier, 'weight_decay': wd}
        learnable_params.append(p_pose)
        
    if train_opts.train_attr == 1:
        assert model.attr_cls is not None
        if model.opts.attr_share_fc == 1:
            params = model.attr_cls.cls.parameters()
        else:
            params = model.attr_cls.parameters()
        wd = train_opts.weight_decay * train_opts.loss_weight_attr if train_opts.adjust_weight_decay == 1 else train_opts.weight_decay
        learnable_params.append({'params': params, 'lr_mult': train_opts.sidetask_lr_multiplier, 'weight_decay': wd})

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

    crit_age = misc.Smooth_Loss(misc.Video_Loss(crit_age))
    meas_age = misc.Video_Age_Analysis()

    crit_pert = misc.L2NormLoss()
    crit_pert = misc.Smooth_Loss(misc.Video_Loss(crit_pert, same_sz = True))

    if train_opts.train_pose == 1:
        crit_pose = misc.Smooth_Loss(nn.MSELoss())
        meas_pose = misc.Pose_MAE(pose_dim = model.opts.pose_dim)

    if train_opts.train_attr == 1:
        crit_attr = misc.Smooth_Loss(nn.BCELoss())
        meas_attr = misc.MeanAP()
        

    ### output training information
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
        fn_snapshot = os.path.join(output_dir, '%s.pth' % epoch)
        print('saving checkpoint to %s' % fn_snapshot)
        model.save_model(fn_snapshot)
        io.save_json(info, fn_info)


    # text_log
    fn_log = os.path.join(output_dir, 'log.txt')
    fout = open(fn_log, 'w')
    print(opts_str, file = fout)

    # pavi_log
    if train_opts.pavi == 1:
        pavi = PaviClient(username = 'ly015', password = '123456')
        pavi.connect(model_name = train_opts.id, info = {'session_text': opts_str})


    # save checkpoint if getting a best performance
    checkbest_name = 'mae_age'
    checkbest_value = sys.float_info.max
    checkbest_epoch = -1

    # main training loop
    epoch = 0

    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for i in xrange(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * optimizer.param_groups[i]['lr_mult']

        if train_opts.lr_decay_pose > 0 :
            p_pose['lr'] = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay_pose)) * p_pose['lr_mult']

        # enable perturb
        if train_opts.pert_enable == 1 and train_opts.pert_start_time == epoch:
            model.pert_opts['enable'] = True
            model.pert_opts['mode'] = train_opts.pert_mode
            model.pert_opts['guide_signal'] = train_opts.pert_guide_signal
            model.pert_opts['guide_index'] = train_opts.pert_guide_index
            model.pert_opts['scale'] = train_opts.pert_scale

            log = 'start feature perturbation at epoch %d' % epoch
            print(log)
            print(log, file = fout)

        # train one epoch
        for batch_idx, age_data in enumerate(age_train_loader):
            optimizer.zero_grad()

            # get age data
            img_seq_age, seq_len_age, age_gt, age_std = age_data
            img_seq_age = Variable(img_seq_age).cuda()
            seq_len_age = Variable(seq_len_age).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_std = age_std.float()
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

            data = [(img_seq_age, seq_len_age), img_pose, img_attr]

            # forward
            age_out, age_fc_out, pose_out, attr_out, pert_out = model.forward_video(data)


            # compute loss
            loss = crit_age(age_fc_out, age_label, seq_len_age) * train_opts.loss_weight_age
            meas_age.add(age_out, age_gt, seq_len_age, age_std)

            if train_opts.train_pose == 1:
                loss += crit_pose(pose_out, pose_gt) * train_opts.loss_weight_pose
                meas_pose.add(pose_out, pose_gt)

            if train_opts.train_attr == 1:
                loss += crit_attr(attr_out, attr_gt) * train_opts.loss_weight_attr
                # don't compute attribute meanAP on training set
                # meas_attr.add(attr_out, attr_gt)

            if model.pert_opts['enable']:
                assert pert_out is not None
                loss += crit_pert(pert_out[0], pert_out[1], seq_len_age) * train_opts.loss_weight_pert
                

            # backward
            loss.backward()

            # optimize
            if train_opts.clip_grad > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), train_opts.clip_grad)
                if total_grad_norm > train_opts.clip_grad:
                    print('Clip gradient: %f ==> %f' % (total_grad_norm, train_opts.clip_grad))

            optimizer.step()

            # display
            if batch_idx % train_opts.display_interval == 0:
                iteration = batch_idx + epoch * len(age_train_loader)

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
                    'iteration': iteration,
                    'epoch': epoch,
                    'loss_age': loss_age, 
                    'mae_age': mae_age
                }

                pavi_outputs = {
                    'loss_age': loss_age,
                    'mae_age_upper': mae_age,
                }

                
                if model.pert_opts['enable']:
                    loss_pert = crit_pert.smooth_loss()
                    crit_pert.clear()

                    log_pert = '[Perturbation]   Loss: %.6f (weight: %.0e)' % (loss_pert, train_opts.loss_weight_pert)
                    log = log + '\n\t' + log_pert
                    
                    train_info['loss_pert'] = loss_pert
                    pavi_outputs['loss_pert'] = loss_pert

                
                if train_opts.train_pose:

                    loss_pose = crit_pose.smooth_loss()
                    mae_pose = meas_pose.mae().tolist()

                    crit_pose.clear()
                    meas_pose.clear()

                    log_pose = '[Pose]   Loss: %.6f (weight: %.0e)  Details:' % (loss_pose, train_opts.loss_weight_pose)

                    for i in range(model.opts.pose_dim):
                        log_pose += '[%s: %.2f]' % (['yaw', 'pitch', 'roll'][i], mae_pose[i])

                    log = log + '\n\t' + log_pose

                    train_info['loss_pose'] = loss_pose
                    train_info['mae_pose'] = mae_pose

                    pavi_outputs['loss_pose'] = loss_pose
                    for i in range(model.opts.pose_dim):
                        pavi_outputs['mae_pose_%s_upper' % ['yaw', 'pitch', 'roll'][i]] = mae_pose[i]



                if train_opts.train_attr:
                    loss_attr = crit_attr.smooth_loss()
                    crit_attr.clear()

                    log_attr = '[Attribute]   Loss: %.6f (weight: %.0e)' % (loss_attr, train_opts.loss_weight_attr)
                    log = log + '\n\t' + log_attr

                    train_info['loss_attr'] = loss_attr
                    pavi_outputs['loss_attr'] = loss_attr


                print(log) # to screen
                print(log, file = fout) # to log file

                info['train_history'].append(train_info)

                if train_opts.pavi == 1:
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)

        # update epoch index
        epoch += 1

        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:
            ## test age, pose, attribute respectively
            iteration = epoch * len(age_train_loader)

            ## set test mode
            model.eval()

            ## test age
            crit_age.clear()
            meas_age.clear()
            crit_pert.clear()

            for batch_idx, age_data in enumerate(age_test_loader):
                
                img_seq_age, seq_len_age, age_gt, age_std = age_data
                img_seq_age = Variable(img_seq_age).cuda()
                seq_len_age = Variable(seq_len_age).cuda()
                age_gt = Variable(age_gt.float()).cuda()
                age_std = age_std.float()
                age_label = age_gt.round().long() - model.opts.min_age

                data = [(img_seq_age, seq_len_age), None, None]

                age_out, age_fc_out, _, _, pert_out = model.forward_video(data)

                loss = crit_age(age_fc_out, age_label, seq_len_age)
                meas_age.add(age_out, age_gt, seq_len_age, age_std)

                if model.pert_opts['enable']:
                    loss = crit_pert(pert_out[0], pert_out[1], seq_len_age)



                print('\r Testing Age %d/%d (%.2f%%)' % (batch_idx, len(age_test_loader), 100.*batch_idx/len(age_test_loader)), end = '')
            print('\n')

            # display
            loss_age = crit_age.smooth_loss()
            mae_age = meas_age.mae()
            ca3 = meas_age.ca(3)
            ca5 = meas_age.ca(5)
            ca10 = meas_age.ca(10)
            lap_err = meas_age.lap_err()
            der = meas_age.stable_der()
            rng = meas_age.stable_range()

            crit_age.clear()
            meas_age.clear()

            log = '[%s] [%s] Test Epoch %d' % (time.ctime(), train_opts.id, epoch)

            log_age = '[Age]   Loss: %.6f (weight: %.0e)   Mae: %.2f\n\tCA(3): %.2f   CA(5): %.2f   CA(10): %.2f   LAP: %.4f\n\tDer: %f   Range: %f' % \
                    (loss_age, train_opts.loss_weight_age, mae_age, ca3, ca5, ca10, lap_err, der, rng)

            log = log + '\n\t' + log_age

            test_info = {
                'iteration': iteration,
                'epoch': epoch,
                'loss_age': loss_age,
                'mae_age': mae_age,
                'ca3': ca3,
                'ca5': ca5,
                'ca10': ca10,
                'lap_err': lap_err,
                'der': der,
                'rng': rng
            }

            pavi_outputs = {
                'loss_age': loss_age,
                'mae_age_upper': mae_age,
                'der_age_upper': der,
            }

            if model.pert_opts['enable']:
                loss_pert = crit_pert.smooth_loss()

                log_pert = '[Perturbation]   Loss: %.6f (weight: %.0e)' % (loss_pert, train_opts.loss_weight_pert)
                log = log + '\n\t' + log_pert

                test_info['loss_pert'] = loss_pert
                pavi_outputs['loss_pert'] = loss_pert

                crit_pert.clear()

            ## test pose
            if train_opts.train_pose:
                crit_pose.clear()
                meas_pose.clear()

                for batch_idx, pose_data in enumerate(pose_test_loader):
                    img_pose, pose_gt = pose_data
                    img_pose = Variable(img_pose).cuda()
                    pose_gt = Variable(pose_gt[:,0:model.opts.pose_dim].float()).cuda()
                    data = [_, img_pose, _]

                    _, _, pose_out, _, _ = model.forward(data)

                    loss = crit_pose(pose_out, pose_gt)
                    meas_pose.add(pose_out, pose_gt)
                    print('\r Testing Pose %d/%d (%.2f%%)' % (batch_idx, len(pose_test_loader), 100.*batch_idx/len(pose_test_loader)), end = '')
                print('\n')

                loss_pose = crit_pose.smooth_loss()
                mae_pose = meas_pose.mae().tolist()

                crit_pose.clear()
                meas_pose.clear()

                log_pose = '[Pose]   Loss: %.6f   Details:' % loss_pose
                for i in range(model.opts.pose_dim):
                        log_pose += '[%s: %.2f]' % (['yaw', 'pitch', 'roll'][i], mae_pose[i])

                log = log + '\n\t' + log_pose

                test_info['loss_pose'] = loss_pose
                test_info['mae_pose'] = mae_pose

                pavi_outputs['loss_pose'] = loss_pose
                for i in range(model.opts.pose_dim):
                    pavi_outputs['mae_pose_%s_upper' % ['yaw', 'pitch', 'roll'][i]] = mae_pose[i]

            ## test attribute
            if train_opts.train_attr:
                crit_attr.clear()
                meas_attr.clear()

                for batch_idx, attr_data in enumerate(attr_test_loader):
                    img_attr, attr_gt = attr_data
                    img_attr = Variable(img_attr).cuda()
                    attr_gt = Variable(attr_gt.float()).cuda()
                    data = [_, _, img_attr]

                    _, _, _, attr_out, _ = model.forward(data)

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

                log = log + '\n\t' + log_attr + '\n\n' + log_attr_details

                test_info['loss_attr'] = loss_attr
                test_info['mean_ap_attr'] = mean_ap_attr.tolist()
                test_info['ap_attr'] = ap_attr.tolist()
                test_info['mean_ap_pn_attr'] = mean_ap_pn_attr.tolist()
                test_info['ap_pn_attr'] = ap_pn_attr.tolist()

                pavi_outputs['loss_attr'] = loss_attr
                pavi_outputs['mean_ap_attr_upper'] = mean_ap_attr.tolist()


            print(log)
            print(log, file = fout)

            info['test_history'].append(test_info)

            if train_opts.pavi == 1:
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


def test_model_video(model, test_opts):
    print('[Joint.test_video] test options: %s' % test_opts)
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()
    model.eval()

    # create dataloader
    test_dset = dataset.load_video_age_dataset(version = test_opts.dataset_version, subset = test_opts.subset, 
        crop_size = test_opts.crop_size, age_rng = [0, 70])
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size = test_opts.batch_size, num_workers = 4)

    # metrics
    meas_age = misc.Video_Age_Analysis()

    age_pred = []
    pose_pred = []
    attr_pred = []

    for batch_idx, data in enumerate(test_loader):

        img_seq, seq_len, age_gt, age_std = data
        img_seq = Variable(img_seq, volatile = True).cuda()
        seq_len = Variable(seq_len, volatile = True).cuda()
        age_gt = age_gt.float()
        age_std = age_std.float()

        # forward
        age_out, _, pose_out, attr_out = model.forward_video_test((img_seq, seq_len))

        meas_age.add(age_out, age_gt, seq_len, age_std)

        if test_opts.output_rst == 1:
            for i, l in enumerate(seq_len):
                l = int(l.data[0])
                age_pred.append(age_out.data.cpu()[i,0:l].numpy().tolist())
                if pose_out is not None:
                    pose_pred.append(pose_out.data.cpu()[i,0:l,:].numpy().tolist())
                else:
                    pose_pred.append(None)
                if attr_out is not None:
                    attr_pred.append(attr_out.data.cpu()[i,0:l,:].numpy().tolist())
                else:
                    attr_pred.append(None)

        print('\rTesting %d/%d (%.2f%%)' % (batch_idx, len(test_loader), 100.*batch_idx/len(test_loader)), end = '')
        sys.stdout.flush()
    print('\n')

    # define output information
    info = {
        'test_opts': vars(test_opts),
        'test_result': {
            'mae': meas_age.mae(),
            'ca3': meas_age.ca(3),
            'ca5': meas_age.ca(5),
            'ca10': meas_age.ca(10),
            'lap_err': meas_age.lap_err(),
            'der': meas_age.stable_der(),
            'range': meas_age.stable_range()
        }
    }

    print('[AgeModel.test_video] Test model id: %s' % test_opts.id)
    for metric in ['mae', 'ca3', 'ca5', 'ca10', 'lap_err', 'der', 'range']:
        print('\t%s\t%f' % (metric, info['test_result'][metric]))

    # output dir
    if test_opts.id.endswith('.pth'):
        # test_opts.id is file name
        output_dir = os.path.dirname(test_opts.id)
    else:
        # test_opts.id is model id
        output_dir = os.path.join('models', test_opts.id)

    assert os.path.isdir(output_dir)

    fn_info = os.path.join(output_dir, 'video_test_info.json')
    io.save_json(info, fn_info)


    if test_opts.output_rst == 1:
        id_lst = test_dset.id_lst
        
        rst = {s_id: {'age': age, 'pose':pose, 'attr': attr} for s_id, age,pose,attr in zip(id_lst, age_pred, pose_pred, attr_pred)}

        fn_rst = os.path.join(output_dir, 'video_age_v%s_test_rst.pkl' % test_opts.dataset_version)
        io.save_data(rst, fn_rst)



if __name__ == '__main__':

    command = opt_parser.parse_command()

    if command == 'train' or command == 'train_video':
        model_opts = opt_parser.parse_opts_joint_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        model = JointModel(opts = model_opts)

        if command == 'train':
            train_model(model, train_opts)
        else:
            train_model_video(model, train_opts)

    elif command == 'finetune' or command == 'finetune_video':
        model_opts = opt_parser.parse_opts_joint_model()
        train_opts = opt_parser.parse_opts_train()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        assert len(train_opts.pre_id) >= 2, 'train_opts.pre_id not set'

        if not train_opts.pre_id[1].endswith('.pth'):
            fn = os.path.join('models', train_opts.pre_id[1], 'final.pth')
        else:
            fn = train_opts.pre_id[1]

        if train_opts.only_load_cnn == 0:
            fn_cnn = None
        else:
            fn_cnn = fn
            fn = None

        model = JointModel(model_opts, fn = fn, fn_cnn = fn_cnn)

        if command == 'finetune':
            train_model(model, train_opts)
        else:
            train_model_video(model, train_opts)

    elif command == 'test' or command == 'test_video':
        test_opts = opt_parser.parse_opts_test()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_opts.gpu_id])

        if test_opts.id.endswith('.pth'):
            fn = test_opts.id
        else:
            fn = os.path.join('models', test_opts.id, 'final.pth')

        model = JointModel(fn = fn)

        if command == 'test':
            test_model(model, test_opts)
        else:
            test_model_video(model, test_opts)

    else:
        raise Exception('invalid command "%s"' % command)