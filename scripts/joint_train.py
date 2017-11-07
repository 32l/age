from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

import util.io as io
from util.pavi import PaviClient
from modules import dataset
from modules import misc
from modules import opt_parser_gan as opt_parser
from modules import resnet
from modules import lib_gan
from modules.gan_model import GANModel

import os
import sys
import numpy as np
from collections import OrderedDict
import time


def finetune_joint(model, train_opts):

    if not train_opts.id.startswith('gan2_joint'):
        train_opts.id = 'gan2_joint' + train_opts.id

    opts_str = opt_parser.opts_to_string([('model_opts', model.opts), ('train_opts', train_opts)])
    print(opts_str)

    ### move model to gpu
    if torch.cuda.device_count() > 1:
        model.cnn = nn.DataParallel(model.cnn)
    model.cuda()

    ### load std decoder
    if model.opts.decoder_id.endswith('.pth'):
        fn_dcd = model.opts.decoder_id
    else:
        fn_dcd = 'models/%s/best.pth'%model.opts.decoder_id

    std_decoder = model.D_net.decoder.__class__(fn = fn_dcd).cuda()

    ### load dataset
    train_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'train',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        split = train_opts.train_split, max_len = train_opts.video_max_len, debug = train_opts.debug)
    test_dset = dataset.load_video_age_dataset(version = train_opts.dataset_version, subset = 'test',
        crop_size = train_opts.crop_size, age_rng = [model.opts.min_age, model.opts.max_age],
        debug = train_opts.debug)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size = train_opts.batch_size, shuffle = True, 
        num_workers = 4, pin_memory = True)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size = torch.cuda.device_count() * 8, 
        num_workers = 4, pin_memory = True)

    
    ### optimizer
    optimizer_age = torch.optim.Adam([
        {'params': model.cnn.parameters(), 'mult': train_opts.cnn_mult},
        {'params': model.age_cls.parameters(), 'mult': train_opts.age_cls_mult}],
        lr = train_opts.lr, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
        eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    optimizer_G = torch.optim.Adam([
        {'params': model.G_net.parameters(), 'mult': train_opts.G_mult}
        ], lr = train_opts.lr, betas = (0.5, train_opts.optim_beta), 
        eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    optimizer_D = torch.optim.Adam([
        {'params': model.D_net.discriminator.parameters(), 'mult': train_opts.D_mult}], lr = train_opts.lr, betas = (0.5, train_opts.optim_beta), 
        eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)

    # train std decoder
    optimizer_dcd = torch.optim.Adam(std_decoder.parameters(), lr = 1e-6, betas = (train_opts.optim_alpha, train_opts.optim_beta), 
        eps = train_opts.optim_epsilon, weight_decay = train_opts.weight_decay)


    # loss function
    crit_age = misc.Ordinal_Hyperplane_Loss(relaxation = model.opts.oh_relaxation, ignore_index = -1)
    crit_age = misc.Smooth_Loss(misc.Video_Loss(crit_age))
    crit_kl = misc.Smooth_Loss(misc.KLLoss())
    meas_age = misc.Video_Age_Analysis()

    crit_G = misc.Smooth_Loss(nn.BCELoss())
    crit_D_R = misc.Smooth_Loss(nn.BCELoss())
    crit_D_F = misc.Smooth_Loss(nn.BCELoss())
    meas_D_R = misc.Smooth_Loss(misc.BlankLoss())
    meas_D_F = misc.Smooth_Loss(misc.BlankLoss())
    meas_acc = misc.Smooth_Loss(misc.BCEAccuracy())

    crit_dcd = misc.Smooth_Loss(nn.L1Loss())
    meas_psnr = misc.Smooth_Loss(misc.PSNR())

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



    # save checkpoint if getting a best performance
    checkbest_name = 'mae'
    checkbest_value = sys.float_info.max
    checkbest_epoch = -1
    checkbest_eof = 1

    ### main training loop
    real_label = 1
    fake_label = 0

    epoch = 0
    while epoch < train_opts.max_epochs:
        # set model mode

        lr = train_opts.lr * (train_opts.lr_decay_rate ** (epoch // train_opts.lr_decay))
        for optimizer in [optimizer_age, optimizer_D, optimizer_age]:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg['mult']

        # train one epoch
        for batch_idx, data in enumerate(train_loader):
            iteration = batch_idx + epoch*len(train_loader)
            model.zero_grad()

            img_pair, seq_len, age_gt, age_std = data
            img_pair = Variable(img_pair).cuda()
            seq_len = Variable(seq_len).cuda()
            age_gt = Variable(age_gt.float()).cuda()
            age_std = age_std.float()

            age_label = age_gt.round().long() - model.opts.min_age

            # train age
            model.eval()
            model.cnn.train()
            model.age_cls.train()

            if train_opts.aug_mode == 'none':
                age_out, fc_out, feat = model.forward_video(img_pair, seq_len)
            else:
                age_out, fc_out, feat = model.forward_video_with_feat_aug(img_pair, seq_len, train_opts)
                seq_len_aug = seq_len.clone()
                seq_len_aug.data.fill_(age_out.size(1))

            loss = crit_age(fc_out, age_label, seq_len)
            meas_age.add(age_out, age_gt, seq_len, age_std)

            loss.backward()
            optimizer_age.step()


            # train std decoder
            model.cnn.eval()
            std_decoder.train()

            img = img_pair[:,0].contiguous().view(-1, 3, 224, 224)
            img_112 = F.avg_pool2d(img, kernel_size = 2)
            # x_mid, x_1 = nn.parallel.data_parallel(model.cnn, img, module_kwargs = {'multi_scale': True})
            x_mid, x_1 = model.cnn.forward(img, multi_scale = True)
            y_112, y_mid = nn.parallel.data_parallel(std_decoder, x_1)

            loss_dcd = crit_dcd(y_112, img_112)
            meas_psnr(std_decoder._detransform(y_112), std_decoder._detransform(img_112))
            loss_dcd.backward()
            optimizer_dcd.step()


            # optmize GAN
            model.eval()
            model.G_net.train()
            model.D_net.train()

            bsz = img_pair.size(0) * 2
            _, _, feat = model.forward_video(img_pair, seq_len)
            feat.detach_()

            feat_in = torch.cat((feat[:,0,:], feat[:,1,:]))
            feat_real = torch.cat((feat[:,1,:], feat[:,0,:]))

            # train D with real
            optimizer_D.zero_grad()

            out = model.D_net(feat_in, feat_real)
            label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()
            
            loss_real = crit_D_R(out, label)
            _ = meas_acc(out, label)
            _ = meas_D_R(out, None)

            loss_real.backward()

            # train D with fake
            noise = Variable(torch.FloatTensor(bsz, model.opts.noise_dim).normal_(0, 1)).cuda()
            feat_res = model.G_net(feat_in, noise)
            feat_fake = feat_in + feat_res

            out = model.D_net(feat_in, feat_fake.detach())
            label = Variable(torch.FloatTensor(bsz, 1).fill_(fake_label)).cuda()
            
            loss_fake = crit_D_F(out, label)
            _ = meas_acc(out, label)
            _ = meas_D_F(out, None)

            loss_fake.backward()
            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()

            out = model.D_net(feat_in, feat_fake)

            label = Variable(torch.FloatTensor(bsz, 1).fill_(real_label)).cuda()
            
            # G loss
            loss_g = crit_G(out, label)

            loss_g.backward()
            optimizer_G.step()

            ### display
            if batch_idx % train_opts.display_interval == 0:

                loss_age = crit_age.smooth_loss(clear = True)
                loss_kl = crit_kl.smooth_loss(clear = True)
                mae = meas_age.mae()
                meas_age.clear()

                loss_dcd = crit_dcd.smooth_loss(clear = True)
                psnr = meas_psnr.smooth_loss(clear = True)

                loss_g = crit_G.smooth_loss(clear = True)
                loss_d = (crit_D_R.smooth_loss(clear = True) + crit_D_F.smooth_loss(clear = True)) / 2
                D_real = meas_D_R.smooth_loss(clear = True)
                D_fake = meas_D_F.smooth_loss(clear = True)
                D_acc = meas_acc.smooth_loss(clear = True)

                log = '[%s] [%s] Train Epoch %d [%d/%d (%.2f%%)]' % \
                    (time.ctime(), train_opts.id, epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset), 100.*batch_idx / len(train_loader))
                log += '\n\tLR: %.3e   loss_age: %.6f    loss_kl: %.6f   mae: %.2f' % (lr, loss_age, loss_kl, mae)
                log += '\n\tloss_G: %.6f   loss_D: %.6f   D_real: %.6f   D_fake: %.6f   GanAcc: %.2f' % (loss_g, loss_d, D_real, D_fake, D_acc)
                log += '\n\tloss_decoder: %.6f   PSNR: %.3f' % (loss_dcd, psnr)

                print(log)
                print(log, file = fout)

                info['train_history'].append({
                    'iteration': iteration,
                    'epoch': epoch,
                    'loss_age': loss_age,
                    'loss_kl': loss_kl,
                    'mae': mae,
                    'loss_g': loss_g,
                    'loss_d': loss_d,
                    'D_real': D_real, 
                    'D_fake': D_fake,
                    'D_acc': D_acc
                    })
                if train_opts.pavi == 1:
                    pavi_outputs = {
                        'loss_age': loss_age,
                        'loss_kl': loss_kl,
                        'mae_age_upper': mae,
                        'loss_g': loss_g,
                        'loss_d': loss_d,
                        'D_real_upper': D_real,
                        'D_fake_upper': D_fake,
                        'D_acc_upper': D_acc
                    }
                    pavi.log(phase = 'train', iter_num = iteration, outputs = pavi_outputs)

        ### update epoch index
        epoch += 1


        # test
        if train_opts.test_interval > 0 and epoch % train_opts.test_interval == 0:

            # set test model
            model.eval()
            std_decoder.eval()

            # clear buffer
            crit_age.clear()
            crit_kl.clear()
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

                crit_age(fc_out, age_label, seq_len)
                crit_kl(fc_out, seq_len)

                meas_age.add(age_out, age_gt, seq_len, age_std)

                print('\rTesting %d/%d (%.2f%%)' % (batch_idx, test_iter, 100.*batch_idx/test_iter), end = '')
                sys.stdout.flush()

                if batch_idx + 1 == test_iter:
                    break
            print('')

            # display
            loss_age = crit_age.smooth_loss(clear = True)
            loss_kl = crit_kl.smooth_loss(clear = True)
            mae = meas_age.mae()
            ca3 = meas_age.ca(3)
            ca5 = meas_age.ca(5)
            ca10 = meas_age.ca(10)
            lap_err = meas_age.lap_err()
            der = meas_age.stable_der()
            rng = meas_age.stable_range()
            meas_age.clear()            


            log = '[%s] [%s] Test Epoch %d   Loss_Age: %.6f   Loss_KL: %.6f   Mae: %.2f\n\tCA(3): %.2f   CA(5): %.2f   CA(10): %.2f   LAP: %.4f\n\tDer: %f   Range: %f' % \
                    (time.ctime(), train_opts.id, epoch, loss_age, loss_kl, mae, ca3, ca5, ca10, lap_err, der, rng)

            print(log)
            print(log, file = fout)

            iteration = epoch * len(train_loader)

            info['test_history'].append({
                    'iteration': iteration,
                    'epoch': epoch, 
                    'loss_age': loss_age, 
                    'loss_kl': loss_kl,
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
                    'loss_age': loss_age,
                    'loss_kl': loss_kl,
                    'mae_age_upper': mae,
                    'der_age_upper': der
                }
                pavi.log(phase = 'test', iter_num = iteration, outputs = pavi_outputs)

            # output verifying image
            vis_loader = torch.utils.data.DataLoader(train_dset, batch_size = 10, shuffle = False)
            for idx, data in enumerate(vis_loader):
                img_pair, seq_len, age, _ = data
                bsz = img_pair.size(0)
                img_pair = Variable(img_pair, volatile = True).cuda()
                seq_len = Variable(seq_len, volatile = True).cuda()
                img_in = F.avg_pool2d(img_pair[:,0], kernel_size = 2)
                _, _, feat_in = model.forward(img_pair[:,0])

                feat_exp = feat_in.unsqueeze(dim = 1).expand(feat_in.size(0), 6, feat_in.size(1)).contiguous().view(-1, feat_in.size(1))
                noise = Variable(torch.FloatTensor(feat_exp.size(0), model.opts.noise_dim).normal_(0, 1)).cuda()
                feat_res = model.G_net(feat_exp, noise)
                feat_fake = feat_exp + feat_res

                dcd_img_in,_ = std_decoder(feat_in.contiguous())
                dcd_img_fake,_  = std_decoder(feat_fake.contiguous())

                out_img = torch.cat((img_in.view(bsz, 1, 3, 112, 112), dcd_img_in.view(bsz, 1, 3, 112, 112), dcd_img_fake.view(bsz, 6, 3, 112, 112)), dim = 1)
                out_img = out_img.view(bsz*8, 3, 112, 112)

                break

            vis_dir = os.path.join(output_dir, 'vis')
            io.mkdir_if_missing(vis_dir)
            fn_vis = os.path.join(vis_dir, '%d.png' % epoch)
            torchvision.utils.save_image(out_img.cpu().data, fn_vis, normalize = True)


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

    train_opts = opt_parser.parse_opts_finetune_joint()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in train_opts.gpu_id])

    if not train_opts.pre_id.endswith('.pth'):
        fn = os.path.join('models', train_opts.pre_id, 'final.pth')
    else:
        fn = train_opts.pre_id

    model = GANModel(fn = fn, load_weight = False)
    if train_opts.load_age_cls == 1:
        model.load_model(fn, modules = ['cnn', 'age_cls', 'G_net', 'D_net'])
    else:
        model.load_model(fn, modules = ['cnn', 'G_net', 'D_net'])
        print('retrain age_cls!')

    finetune_joint(model, train_opts)