#  parse user-input options

import argparse


def parse_command():
    parser = argparse.ArgumentParser()

    parser.add_argument('command', type = str, default = 'help',
        choices = ['train', 'test', 'finetune', 'help'], help = 'valid commands: train, test, help')


    command = parser.parse_known_args()[0].command

    return command


def parse_opts_age_model():

    parser = argparse.ArgumentParser()

    parser.add_argument('--cnn', type = str, default = 'resnet18', choices = ['resnet18', 'resnet50', 'vgg16'],
        help = 'cnn network')

    parser.add_argument('--min_age', type = int, default = 0,
        help = 'min age')

    parser.add_argument('--max_age', type = int, default = 100,
        help = 'max age')

    parser.add_argument('--num_fc', type = int, default = 1, choices = [1, 2],
        help = 'number of fc layers in classifier')

    parser.add_argument('--fc_sizes', type = int, default = [128],
        help = 'size of intermediate fc layers')

    parser.add_argument('--cls_type', type = str, default = 'oh', choices = ['oh', 'dex'],
        help = 'oh: ordinal hyperplane; dex: deep expectation')

    parser.add_argument('--oh_relaxation', type = int, default = 3,
        help = 'relaxation parameter of ordinal hyperplane loss')

    parser.add_argument('--dropout', type = float, default = 0.5,
        help = 'dropout rate')

    opts = parser.parse_known_args()[0]
    return opts


def parse_opts_train():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--id', type = str, default = 'default',
        help = 'model id')

    parser.add_argument('--gpu_id', type = int, default = [0], nargs = '*',
        help = 'GPU device id used for model training')

    parser.add_argument('--debug', type = int, default = 0,
        help = 'use a small set of data to debug model')
    

    # data
    parser.add_argument('--dataset', type = str, default = 'imdb_wiki',
        choices = ['imdb_wiki', 'imdb_wiki_good', 'megaage', 'morph'],
        help = 'dataset name [imdb_wiki|imdb_wiki_good|megaage|morph]')

    parser.add_argument('--face_alignment', type = str, default = '21',
        choices = ['3', '21', 'none'],
        help = 'face alignment mode. see prepro_general for more information')

    parser.add_argument('--crop_size', type = int, default = 128,
        help = 'center crop size')

    # parser.add_argument()


    # optimization
    parser.add_argument('--max_epochs', type = int, default = 30,
        help = 'number of training epochs')

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'batch size')

    parser.add_argument('--optim', type = str, default = 'sgd',
        choices = ['sgd', 'adam'])

    parser.add_argument('--lr', type = float, default = 1e-3,
        help = 'learning rate')

    parser.add_argument('--lr_decay', type = int, default = 10,
        help = 'every how many epochs does the learning rate decay')

    parser.add_argument('--lr_decay_rate', type = float, default = 0.1,
        help = 'learning decay rate')

    parser.add_argument('--weight_decay', type = float, default = 5e-4,
        help = 'L2 weight decay')

    parser.add_argument('--momentum', type = float, default = 0.9,
        help = 'momentum for SGD')

    parser.add_argument('--optim_alpha', type = float, default = 0.9,
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

    parser.add_argument('--snapshot_interval', type = int, default = 10,
        help = 'every how many epochs save model parameters to file')

    parser.add_argument('--average_loss', type = int, default = -1, 
        help = 'average the last n loss when display')

    parser.add_argument('--cls_lr_multiplier', type = float, default = 10,
        help = 'learning rate multiplier of the classifier layers')

    parser.add_argument('--loss_sample_normalize', type = int, default = 1,
        help = 'for oh loss, 1 means averaging loss over samples, 0 means averaging loss over channels')


    # finetune
    parser.add_argument('--pre_id', type = str, default = [], nargs = '*',
        help = 'the list of pretrained model IDs (or model files if end with ".pth")')

    parser.add_argument('--only_load_cnn', type = int, default = 1, choices = [0, 1],
        help = '0-load pretrained weights for all layers, 1-load pretrained weights only for CNN')


    opts = parser.parse_known_args()[0]
    return opts

def parse_opts_test():

    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--id', type = str, default = None,
        help = 'model id or model file if end with .pth')

    parser.add_argument('--gpu_id', type = int, default = [0], nargs = '*',
        help = 'GPU device id used for model training')

    parser.add_argument('--output_rst', type = int, default = 0, choices = [0, 1],
        help = 'output predicted age of each sample to a pkl file')


    # data
    parser.add_argument('--dataset', type = str, default = 'imdb_wiki',
        choices = ['imdb_wiki', 'imdb_wiki_good', 'megaage', 'morph'],
        help = 'dataset name [imdb_wiki|imdb_wiki_good|megaage|morph]')

    parser.add_argument('--subset', type = str, default = 'test',
        choices = ['train', 'test', 'val'],
        help = 'subset of the dataset')

    parser.add_argument('--face_alignment', type = str, default = '21',
        choices = ['3', '21', 'none'],
        help = 'face alignment mode. see prepro_general for more information')

    parser.add_argument('--batch_size', type = int, default = 512,
        help = 'batch size')

    parser.add_argument('--crop_size', type = int, default = 128,
        help = 'center crop size')

    opts = parser.parse_known_args()[0]
    return opts
    

if __name__ == '__main__':

    # test parse_command()

    command = parse_command()

    print(command)