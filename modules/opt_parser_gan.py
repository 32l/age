#  parse user-input options

import argparse

def opts_to_string(opts_lst):

    opts_str = ''
    for opt_name, opt in opts_lst:
        if not isinstance(opt, dict):
            opt = vars(opt)
        opts_str += (opt_name + '\n')
        opts_str += '\n'.join(['  %-20s: %s' % (k,v) for k,v in opt.iteritems()])
        opts_str += '\n\n'
    return opts_str

def update_opts_from_dict(opts, opts_dict, exceptions = []):
    
    for k, v in opts_dict.iteritems():
        if k in exceptions:
            continue
        
        # assert k in opts, '%s not in namespace' % k
        if k not in opts:
            print('%s = %s been ignored' % (k,v))
        else:
            opts.__dict__[k] = v
    
    return opts


def parse_command():
    parser = argparse.ArgumentParser()

    parser.add_argument('command', type = str, default = 'help',
        choices = ['pretrain', 'train', 'train_gan', 'pretrain_gan', 'finetune_fix',
        'retrain', 'show_feat', 'extract_feat'])

    command = parser.parse_known_args()[0].command

    return command


def parse_opts_gan_model(args = None, namespace = None):
    '''
    Parse argments for creating GANModel
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--cnn', type = str, default = 'resnet18', choices = ['resnet18', 'resnet50', 'vgg16'],
        help = 'cnn network')

    # age model
    parser.add_argument('--min_age', type = int, default = 0,
        help = 'min age')

    parser.add_argument('--max_age', type = int, default = 70,
        help = 'max age')

    parser.add_argument('--age_fc_size', type = int, default = 128,
        help = '[Deprecated] size of middle fc layer in age classifier')

    parser.add_argument('--cls_type', type = str, default = 'oh', choices = ['oh', 'dex'],
        help = 'oh: ordinal hyperplane; dex: deep expectation')

    parser.add_argument('--oh_relaxation', type = int, default = 3,
        help = 'relaxation parameter of ordinal hyperplane loss')

    parser.add_argument('--dropout', type = float, default = 0,
        help = 'dropout rate')

    parser.add_argument('--age_cls_hidden', type = int, default = [128,128], nargs = '*',
        help = 'dim of fc-layers in age_cls')

    # GAN
    parser.add_argument('--noise_dim', type = int, default = 100,
        help = 'noise signal dimension')

    parser.add_argument('--G_hidden', type = int, default = [128, 32, 128], nargs = '*',
        help = 'latent space dimenssion of generator')

    parser.add_argument('--G_nonlinear', type = str, default = 'elu',
        choices = ['elu', 'lrelu'],
        help = 'nonlinear layer type of G_net')

    parser.add_argument('--input_relu', type = int, default = 0,
        help = 'relu input feature')

    parser.add_argument('--D_hidden2', type = int, default = [128], nargs = '*',
        help = 'latent space dimentions of discriminator')

    parser.add_argument('--D_hidden', type = int, default = [], nargs = '*',
        help = 'latent space dimentions of discriminator')

    parser.add_argument('--D_type', type = str, default = 'naive',
        choices = ['naive', 'm', 'cm', 'd'],
        help = 'naive: simple D; m: matching D; cm: cinditional matching D')

    parser.add_argument('--decoder_id', type = str, default = 'dcd_3.2')



    opts = parser.parse_known_args(args = args, namespace = namespace)[0]

    return opts


def parse_opts_test():
    '''
    Parse options for test model
    '''
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--id', type = str, default = 'default',
        help = 'model id')
    
    parser.add_argument('--gpu_id', type = int, default = [0], nargs = '*',
        help = 'GPU device id used for testing')

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'batch size')

    
    
    opts = parser.parse_known_args()[0]
    
    return opts

    
def parse_opts_retrain():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--id', type = str, default = 'default',
        help = 'ID of model to retrain')
    
    parser.add_argument('--mode', type = str, default = 'train_gan',
        choices = ['pretrain', 'pretrain_gan', 'train_gan', 'finetune_fix'],
        help = 'training mode ')

    opts = parser.parse_known_args()[0]

    return opts

def basic_train_opts_parser():
    '''
    Return a parser of basic training options, which can be used as a parent for parsers of specific training phases.
    '''

    parser = argparse.ArgumentParser(add_help = False)

    # misc
    parser.add_argument('--id', type = str, default = 'default',
        help = 'model id')

    parser.add_argument('--gpu_id', type = int, default = [0], nargs = '*',
        help = 'GPU device id used for model training')

    parser.add_argument('--debug', type = int, default = 0,
        help = 'use a small set of data to debug model')

    parser.add_argument('--pavi', type = int, default = 1, choices = [0, 1],
        help = 'use pavi log')


    # data
    parser.add_argument('--dataset', type = str, default = 'video_age',
        choices = ['imdb_wiki', 'imdb_wiki_good', 'megaage', 'morph', 'lap', 'video_age'],
        help = 'dataset name [imdb_wiki|imdb_wiki_good|megaage|morph|lap|video_age]')

    parser.add_argument('--face_alignment', type = str, default = '21',
        choices = ['3', '21', 'none'],
        help = 'face alignment mode. see prepro_general for more information')

    parser.add_argument('--crop_size', type = int, default = 128,
        help = 'center crop size')

    # for video age dataset
    parser.add_argument('--dataset_version', type = str, default = '2.0',
        choices = ['1.0', '2.0'],
        help = 'video_age dataset version')

    parser.add_argument('--train_split', type = str, default = '',
        choices = ['0.1', '0.2', '0.5', ''],
        help = 'video_age dataset training split')

    parser.add_argument('--video_max_len', type = int, default = 2,
        help = 'max frame number in each video sample')


    # optimization
    parser.add_argument('--max_epochs', type = int, default = 30,
        help = 'number of training epochs')

    parser.add_argument('--batch_size', type = int, default = 32,
        help = 'batch size')

    parser.add_argument('--optim', type = str, default = 'adam',
        choices = ['sgd', 'adam'])

    parser.add_argument('--lr', type = float, default = 1e-4,
        help = 'learning rate')

    parser.add_argument('--lr_decay', type = int, default = 20,
        help = 'every how many epochs does the learning rate decay')

    parser.add_argument('--lr_decay_rate', type = float, default = 0.1,
        help = 'learning decay rate')

    parser.add_argument('--weight_decay', type = float, default = 0.0005,
        help = 'L2 weight decay')

    parser.add_argument('--momentum', type = float, default = 0.9,
        help = 'momentum for SGD')

    parser.add_argument('--optim_alpha', type = float, default = 0.5,
        help = 'alpha for adam')

    parser.add_argument('--optim_beta', type = float, default = 0.999,
        help = 'beta for adam')

    parser.add_argument('--optim_epsilon', type = float, default = 1e-8,
        help = 'epsilon that goes into denominator for smoothing')

    parser.add_argument('--display_interval', type = int, default = 10,
        help = 'every how many batchs display training loss')

    parser.add_argument('--test_interval', type = int, default = 1,
        help = 'every how many epochs do test')

    parser.add_argument('--test_iter', type = int, default = -1,
        help = 'test iterations. -1 means useing all samples in test set')

    parser.add_argument('--snapshot_interval', type = int, default = -1,
        help = 'every how many epochs save model parameters to file')

    parser.add_argument('--kl_loss_weight', type = float, default = 0.)


    return parser


def parse_opts_pretrain():

    parser = argparse.ArgumentParser(parents = [basic_train_opts_parser()])

    parser.add_argument('--pre_id', type = str, default = 'models/age_pre_2.2/9.pth',
        help = 'ID of pretrained model on IMDB-WIKI data')

    parser.add_argument('--age_cls_lr_mult', type = float, default = 10.0,
        help = 'learning rate multiplier of age_cls module')

    parser.add_argument('--age_cls_only', type = int, default = 0, choices = [0,1],
        help = 'fix cnn parameters and only train age_cls')


    opts = parser.parse_known_args()[0]

    return opts

def parse_opts_train_gan():

    parser = argparse.ArgumentParser(parents = [basic_train_opts_parser()])

    parser.add_argument('--pre_id', type = str, default = 'gan2_pre_1.6',
        help = 'ID of pretrained model on age datasets')
        
    parser.add_argument('--gan_pretrained', type = int, default = 0,
        help = 'set 1 if the GAN in pretrained model has been pretrained')

    parser.add_argument('--D_lr_mult', type = float, default = 1.0,
        help = 'learning rate multiplier for D (should be <= 1.0)')

    parser.add_argument('--G_lr_mult', type = float, default = 1.0,
        help = 'learning rate multiplier for G (should be <= 1.0)')

    parser.add_argument('--G_l2_weight', type = float, default = 0.0,
        help = 'L2-norm on generated feat_res')

    parser.add_argument('--D_iter', type = int, default = 1,
        help = 'number of D iterations every G iteration')

    parser.add_argument('--real_label', type = float, default = 1.0,
        help = 'value of real label')

    opts = parser.parse_known_args()[0]

    return opts


def parse_opts_finetune_fix_cnn():
    
    parser = argparse.ArgumentParser(parents = [basic_train_opts_parser()])
    
    parser.add_argument('--pre_id', type = str, default = 'models/gan2_2.0/final.pth',
        help = 'ID of pretrained model (both CNN and GAN should be trained)')
    
    parser.add_argument('--aug_mode', type = str, default = 'gan',
        choices = ['gan', 'gaussian', 'none'],
        help = 'augmentation method')
    
    parser.add_argument('--aug_scale', type = float, default = 1.,
        help = 'augmentation scale. for guassian noise, it is the std')
    
    parser.add_argument('--aug_rate', type = int, default = 5,
        help = 'number of augmented samples for each real sample')
    
    parser.add_argument('--aug_pure', type = int, default = 0,
        help = 'using augmented data only')

    parser.add_argument('--load_age_cls', type = int, default = 0,
        help = 'load age_cls from pretrained model or retrain it')

    opts = parser.parse_known_args()[0]
    
    return opts

def parse_opts_finetune_joint():
    
    parser = argparse.ArgumentParser(parents = [basic_train_opts_parser()])
    
    parser.add_argument('--pre_id', type = str, default = 'models/gan2_8.2/final.pth',
        help = 'ID of pretrained model (both CNN and GAN should be trained)')

    parser.add_argument('--aug_mode', type = str, default = 'gan',
        choices = ['gan', 'gaussian', 'none'],
        help = 'augmentation method')
    
    parser.add_argument('--aug_scale', type = float, default = 1.,
        help = 'augmentation scale. for guassian noise, it is the std')
    
    parser.add_argument('--aug_rate', type = int, default = 5,
        help = 'number of augmented samples for each real sample')
    
    parser.add_argument('--aug_pure', type = int, default = 0,
        help = 'using augmented data only')

    parser.add_argument('--load_age_cls', type = int, default = 0,
        help = 'load age_cls from pretrained model or retrain it')



    parser.add_argument('--cnn_mult', type = float, default = 0.1)
    parser.add_argument('--age_cls_mult', type = float, default = 1.0)
    parser.add_argument('--G_mult', type = float, default = 0.1)
    parser.add_argument('--D_mult', type = float, default = 0.02)

    opts = parser.parse_known_args()[0]
    
    return opts
    
    
def parse_opts_decoder(args = None, namespace = None):

    parser = argparse.ArgumentParser()
    

    parser.add_argument('--block_type', type = str, default = 'dconv', 
        choices = ['dconv', 'conv', 'pixel', 'mix'])

    opts = parser.parse_known_args(args = args, namespace = namespace)[0]
    return opts

def parse_opts_train_decoder():

    parser = argparse.ArgumentParser(parents = [basic_train_opts_parser()])

    parser.add_argument('--cnn_id', type = str, default = 'gan2_pre_1.6',
        help = 'ID of gan_model which provides CNN encoder')

    parser.add_argument('--mid_loss_weight', type = float, default = 0.1,
        help = 'loss weight of intermediate leverl feature map')

    opts = parser.parse_known_args()[0]
    
    return opts

