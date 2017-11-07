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
import resnet

import os
import sys
import numpy as np
from collections import OrderedDict
import time

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if 'weight' in m._parameters:
            m.weight.data.normal_(0.0, 0.02)
        if 'bias' in m._parameters and m.bias is not None:
            m.bias.data.fill_(0.0)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)





class DconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DconvBlock, self).__init__()

        self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
        Input:
            x: (bsz, in_channels, w_in, h_in)
        Ooutput:
            y: (bsz, out_channels, w_out, h_out)
            w_out = (w_in * stride + padding
        '''
        y = self.dconv(x)
        y = self.bn(y)

        return y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.upsample(x)
        y = self.conv(y)
        y = self.bn(y)
        return y


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelShuffleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.pixel_shuffle(y)
        y = self.bn(y)
        return y

class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixBlock, self).__init__()
        self.dconv_block = DconvBlock(in_channels, out_channels)
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pixel_shuffle_block = PixelShuffleBlock(in_channels, out_channels)

    def forward(self, x):

        return (self.dconv_block(x) + self.conv_block(x) + self.pixel_shuffle_block(x)) / 3.0

    

class DecoderModel(nn.Module):

    def _update_opts(self, opts):
        '''
        update old version opts
        '''
        opts = opt_parser.parse_opts_decoder(namespace = opts)
        return opts

    def __init__(self, opts = None, fn = None, load_weight = True):

        assert (opts or fn), 'Error: either "opts" or "fn" should be provided'
        super(DecoderModel, self).__init__()

        if opts is None:
            opts = torch.load(fn, map_location=lambda storage, loc: storage)['opts']
            opts = self._update_opts(opts)
        self.opts = opts

        # crate decoder
        if opts.block_type == 'dconv':
            Block = DconvBlock
        elif opts.block_type == 'conv':
            Block = ConvBlock
        elif opts.block_type == 'pixel':
            Block = PixelShuffleBlock
        elif opts.block_type == 'mix':
            Block = MixBlock

        self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 7, bias = False),
                nn.BatchNorm2d(512))
        self.relu1 = nn.LeakyReLU(0.2, inplace = False)
        self.layer2 = Block(in_channels = 512, out_channels = 256)
        self.relu2 = nn.LeakyReLU(0.2, inplace = False)
        self.layer3 = Block(in_channels = 256, out_channels = 128)
        self.relu3 = nn.LeakyReLU(0.2, inplace = False)
        self.layer4 = Block(in_channels = 128, out_channels = 64)
        self.relu4 = nn.LeakyReLU(0.2, inplace = False)
        self.layer5 = Block(in_channels = 64, out_channels = 32)
        self.relu5 = nn.LeakyReLU(0.2, inplace = False)
        self.layer6 = nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias = True)
        
        # DeTransformer
        # self.detransform = torchvision.transforms.Normalize([-.485/.229, -.456/.224, -.406/.225], [1/.229, 1/.224, 1/.225])

        if fn and load_weight:
            self.load_model(fn)
        else:
            self.apply(weights_init)

    def _detransform(self, x):
        assert x.dim() == 4
        x = x.clone()
        x[:,0] = x[:,0] * .229 + .485
        x[:,1] = x[:,1] * .224 + .456
        x[:,2] = x[:,2] * .225 + .406

        return x


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
            'state_dict': self._get_state_dict(self),
        }
        torch.save(model_info, fn)

    def load_model(self, fn, modules = None):

        model_info = torch.load(fn, map_location=lambda storage, loc: storage)
        if modules == None:
            self.load_state_dict(model_info['state_dict'])
            return
        
        # for m_name in modules:
        #     self.__getattr__(m_name).load_state_dict(model_info['%s_state_dict' % m_name])
        #     print('[GANModel.load_model] %s <= %s' % (m_name, fn))

    def forward(self, x):
        '''
        Input:
            x: (bsz, feat_size)
        Output:
            y_7x7: (bsz, 512, 7, 7)
            y_14x14: (bsz, 256, 14, 14)
            y_28x28: (bsz, 128, 28, 28)
            y_56x56: (bsz, 64, 56, 56)
            y_112x112: (bsz, 3, 128, 128)
        '''
        bsz = x.size(0)

        y_1x1 = x.view(x.size(0), x.size(1), 1, 1)
        y_7x7 = self.layer1(y_1x1)
        y_14x14 = self.layer2(self.relu1(y_7x7))
        y_28x28 = self.layer3(self.relu2(y_14x14))
        y_56x56 = self.layer4(self.relu3(y_28x28))
        
        y_112x112 = self.layer5(self.relu4(y_56x56))
        y_112x112 = self.layer6(self.relu5(y_112x112))

        return y_112x112, (y_56x56, y_28x28, y_14x14, y_7x7)


def train_model(model, train_opts, cnn):
    if not train_opts.id.startswith('dcd_'):
        train_opts.id = 'dcd_' + train_opts.id

    opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to GPU
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     cnn = nn.DataParallel(cnn)

    #     model.opts = model.module.opts
    #     model._detransform = model.module._detransform

    model.cuda()
    cnn.cuda()
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad = False

    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [0,70],
        split = train_opts.train_split, mode = 'image', debug = train_opts.debug)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [0,70],
        mode = 'image', debug = train_opts.debug)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = 48 * torch.cuda.device_count(), 
        num_workers = 4, pin_memory = True)


    ### create optimizer
    if train_opts.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = train_opts.lr, weight_decay = train_opts.weight_decay, momentum = train_opts.momentum)

    elif train_opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
            eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    ### loss function

    crit_img = misc.Smooth_Loss(nn.L1Loss())
    crit_mid = [
        ('7x7', misc.Smooth_Loss(nn.MSELoss())),
        ('14x14', misc.Smooth_Loss(nn.MSELoss())),
        ('28x28', misc.Smooth_Loss(nn.MSELoss())),
        ('56x56', misc.Smooth_Loss(nn.MSELoss()))
    ]
    meas_psnr = misc.Smooth_Loss(misc.PSNR())

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
    checkbest_name = 'psnr'
    checkbest_value = sys.float_info.max
    checkbest_epoch = -1
    checkbest_eof = -1



    ### main training loop
    epoch = 0
    
    while epoch < train_opts.max_epochs:

        # set model mode
        model.train()

        # update learning rate
        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # train one epoch
        for batch_idx, data in enumerate(train_loader):

            model.zero_grad()

            img, _, _ = data
            img = Variable(img).cuda()
            img_112x112 = F.avg_pool2d(img, kernel_size = 2)

            # cnn forward
            # x_56x56, x_28x28, x_14x14, x_7x7, x_1x1 = cnn.forward_multi_scale(img)
            # y_112x112, (y_56x56, y_28x28, y_14x14, y_7x7) = model(x_1x1)

            # x_mid, x_1x1 = cnn.forward(img, multi_scale = True)
            # y_112x112, y_mid = model(x_1x1)

            x_mid, x_1x1 = nn.parallel.data_parallel(cnn, img, module_kwargs = {'multi_scale': True})
            y_112x112, y_mid = nn.parallel.data_parallel(model, x_1x1)

            loss = crit_img(y_112x112, img_112x112)
            for x, y, (l, crit) in zip(x_mid, y_mid, crit_mid):
                loss += crit(y, x) * train_opts.mid_loss_weight
            
            meas_psnr(model._detransform(y_112x112), model._detransform(img_112x112))

            loss.backward()
            # optimize
            optimizer.step()

            # display
            if batch_idx % train_opts.display_interval == 0:

                loss_lst =[('img', crit_img.smooth_loss(clear = True))]
                psnr = meas_psnr.smooth_loss(clear = True)
                
                for l, crit in crit_mid:
                    loss_lst.append((l, crit.smooth_loss(clear = True)))

                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)]   LR: %.3e' %\
                        (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size,
                        len(train_loader.dataset), 100. * batch_idx / len(train_loader),lr)

                log += '\n\t' + '  '.join(['Loss_%s: %.3f' % (l, loss) for l, loss in loss_lst]) + ' PSNR: %.3f' % psnr
                

                print(log) # to screen
                print(log, file = fout) # to log file

                iteration = batch_idx + epoch * len(train_loader)
                info['train_history'].append({
                    'iteration': iteration,
                    'epoch': epoch, 
                    'loss_img': loss_lst[0][1],
                    'loss_mid': sum([l for _, l in loss_lst[1::]]),
                    'psnr': psnr
                    })

                if train_opts.pavi == 1:
                    pavi_outputs = {
                        'loss_img': loss_lst[0][1],
                        'loss_mid': sum([l for _, l in loss_lst[1::]]),
                        'psnr_upper': psnr
                    }
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)

        # update epoch index
        epoch += 1

        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test model
            model.eval()

            # clear buffer
            crit_img.clear()
            for _, crit in crit_mid:
                crit.clear()

            # set test iteration
            test_iter = train_opts.test_iter if train_opts.test_iter > 0 else len(test_loader)    

            vis_real = []
            vis_fake = []
            for batch_idx, data in enumerate(test_loader):
                img, _, _ = data
                img = Variable(img, volatile = True).cuda()
                img_112x112 = F.avg_pool2d(img, kernel_size = 2)

                # cnn forward
                # x_56x56, x_28x28, x_14x14, x_7x7, x_1x1 = cnn.forward_multi_scale(img)
                # y_112x112, (y_56x56, y_28x28, y_14x14, y_7x7) = model(x_1x1)

                x_mid, x_1x1 = nn.parallel.data_parallel(cnn, img, module_kwargs = {'multi_scale': True})
                y_112x112, y_mid = nn.parallel.data_parallel(model, x_1x1)

                crit_img(y_112x112, img_112x112)
                meas_psnr(model._detransform(y_112x112), model._detransform(img_112x112))

                for x, y, (l, crit) in zip(x_mid, y_mid, crit_mid):
                    crit(y, x)

                vis_real.append(img_112x112.data.cpu()[0])
                vis_fake.append(y_112x112.data.cpu()[0])

                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()
                if batch_idx + 1 == test_iter:
                    break
            print('')

            # visualize
            num_vis = min(40, len(vis_real))
            vis_real = torch.stack(vis_real)[0:num_vis]
            vis_fake = torch.stack(vis_fake)[0:num_vis]
            vis_dir = os.path.join(output_dir, 'vis')
            io.mkdir_if_missing(vis_dir)
            
            torchvision.utils.save_image(vis_real, os.path.join(vis_dir, 'real.png'), normalize = True)
            torchvision.utils.save_image(vis_fake, os.path.join(vis_dir, 'fake_%d.png'%epoch), normalize = True)

            # display
            loss_lst =[('img', crit_img.smooth_loss(clear = True))]
            for l, crit in crit_mid:
                loss_lst.append((l, crit.smooth_loss(clear = True)))
            psnr = meas_psnr.smooth_loss(clear = True)


            log = '=== [%s] [%s] Test Epoch %d ===' % (time.ctime(), train_opts.id, epoch)
            log += '\n\t' + '  '.join(['Loss_%s: %.3f' % (l, loss) for l, loss in loss_lst]) + ' PSNR: %.3f' % psnr

            print(log)
            print(log, file = fout)

            iteration = epoch * len(train_loader)

            info['test_history'].append({
                    'iteration': iteration,
                    'epoch': epoch, 
                    'loss_img': loss_lst[0][1], 
                    'loss_mid': sum([l for _, l in loss_lst[1::]]),
                    'psnr': psnr
                    })

            if train_opts.pavi == 1:
                pavi_outputs = {
                    'loss_img': loss_lst[0][1],
                    'loss_mid': sum([l for _, l in loss_lst[1::]]),
                    'psnr_upper': psnr
                }
                pavi.log(phase = 'test', iter_num = iteration, outputs = pavi_outputs)


            # snapshot best
            if info['test_history'][-1][checkbest_name] * checkbest_eof < checkbest_value:
                checkbest_value = info['test_history'][-1][checkbest_name] * checkbest_eof
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

    if command == 'train':

        model_opts = opt_parser.parse_opts_decoder()
        train_opts = opt_parser.parse_opts_train_decoder()

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

        # load cnn
        if train_opts.cnn_id.endswith('.pth'):
            cnn_fn = train_opts.cnn_id
        else:
            cnn_fn = 'models/%s/best.pth' % train_opts.cnn_id            
        cnn_info = torch.load(cnn_fn, map_location=lambda storage, loc: storage)

        cnn = resnet.resnet18()
        cnn.load_state_dict(cnn_info['cnn_state_dict'])

        model = DecoderModel(opts = model_opts)
        train_model(model, train_opts, cnn)
    else:
        raise Exception('invalid command "%s"' % command)

