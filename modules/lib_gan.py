from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from collections import OrderedDict


class Generator(nn.Module):
    '''
    Create Generator.
    '''
    def __init__(self, opts):
        
        super(Generator, self).__init__()
        
        cnn_feat_map = {'resnet18': 512, 'resnet50': 2048, 'vgg16': 2048}
        self.cnn_feat_size = cnn_feat_map[opts.cnn]
        self.noise_dim = opts.noise_dim
        
        hidden_lst = [self.cnn_feat_size + self.noise_dim] + opts.G_hidden + [self.cnn_feat_size]
        layers = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(hidden_lst, hidden_lst[1::])):
            layers['fc%d' % n] = nn.Linear(dim_in, dim_out, bias = False)
            if n < len(hidden_lst) - 2:
                layers['bn%d' % n] = nn.BatchNorm1d(dim_out)
                layers['leaky_relu%d'%n] = nn.LeakyReLU(0.2)
                #layers['elu%d' % n] = nn.ELU()
        
        self.net = nn.Sequential(layers)
    
    def forward(self, feat_in, noise):
        '''
        Input:
            feat_in (bsz, cnn_feat_size): input feature (not ReLUed)
            noise (bsz, noise_dim): input gaussian noise
        Output:
            feat_res (bsz, cnn_feat_size): generated feature residual
            
        '''
        
        x = torch.cat((feat_in, noise), dim = 1)
        return self.net(x)
        

class Discriminator(nn.Module):
    '''
    Bisic Discriminator without condition and classification, which takes the feature of
    a frame as input and output the probability of real frame.
    '''
    def __init__(self, opts):
        
        super(Discriminator, self).__init__()

        cnn_feat_map = {'resnet18': 512, 'resnet50': 2048, 'vgg16': 2048}
        self.cnn_feat_size = cnn_feat_map[opts.cnn]
        
        hidden_lst = [self.cnn_feat_size] + opts.D_hidden + [1]
        layers = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(hidden_lst, hidden_lst[1::])):
            layers['fc%d' % n] = nn.Linear(dim_in, dim_out, bias = False)
            if n < len(hidden_lst) - 2:
                layers['bn%d' % n] = nn.BatchNorm1d(dim_out)
                layers['leaky_relu%d' % n] = nn.LeakyReLU(0.2)
        layers['sigmoid'] = nn.Sigmoid()
        
        self.net = nn.Sequential(layers)
    
    def forward(self, feat_in):
        
        return self.net(feat_in)

        
class M_Discriminator(nn.Module):
    '''
    Matching Disriminator. The input is two features from reference frame (real) and target frame (generated or real),
    The output is the probability that the target frame feature is real.
    '''
    def __init__(self, opts):
        
        super(M_Discriminator, self).__init__()
        
        cnn_feat_map = {'resnet18': 512, 'resnet50': 2048, 'vgg16': 2048}
        self.cnn_feat_size = cnn_feat_map[opts.cnn]
        
        # net1: parallel net
        hidden_lst1 = [self.cnn_feat_size] + opts.D_hidden
        layers1 = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(hidden_lst1, hidden_lst1[1::])):
            layers1['fc%d' % n] = nn.Linear(dim_in, dim_out, bias = False)
            layers1['bn%d' % n] = nn.BatchNorm1d(dim_out)
            layers1['leaky_relu%d' % n] = nn.LeakyReLU(0.2)
        
        self.net1 = nn.Sequential(layers1)
        
        # net2: fusing net
        hidden_lst2 = [2 * hidden_lst1[-1]] + opts.D_hidden2 + [1]
        layers2 = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(hidden_lst2, hidden_lst2[1::])):
            layers2['fc%d' % n] = nn.Linear(dim_in, dim_out, bias = False)
            if n < len(hidden_lst2) - 2:
                layers2['bn%d' % n] = nn.BatchNorm1d(dim_out)
                layers2['leaky_relu%d' % n] = nn.LeakyReLU(0.2)
        layers2['sigmoid'] = nn.Sigmoid()
        
        self.net2 = nn.Sequential(layers2)
    
    def forward(self, feat_ref, feat_tar):
        
        bsz, feat_size = feat_ref.size()
        
        x = torch.cat((feat_ref, feat_tar), dim = 0)
        x = self.net1(x)
        
        y = torch.cat((x[0:bsz,:], x[bsz::,:]), dim = 1).contiguous()
        
        return self.net2(y)

class CM_Discriminator(nn.Module):
    '''
    Conditional Matching Discriminator. The input is the features of two frames.
    There are N possible matching type between the input frames.
    The output is N+1 probabilities, corresponding to the N matching type and an extra
    situation where the second frame is generated.
    '''
    
    def __init__(self, opts):
        
        super(CM_Discriminator, self).__init__()
        
        cnn_feat_map = {'resnet18': 512, 'resnet50': 2048, 'vgg16': 2048}
        self.cnn_feat_size = cnn_feat_map[opts.cnn]
        self.num_cls = opts.D_num_cls
        
        # net1: parallel net
        hidden_lst1 = [self.cnn_feat_size] + opts.D_hidden
        layers1 = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(hidden_lst1, hidden_lst1[1::])):
            layers1['fc%d' % n] = nn.Linear(dim_in, dim_out, bias = False)
            layers1['bn%d' % n] = nn.BatchNorm1d(dim_out)
            layers1['leaky_relu%d' % n] = nn.LeakyReLU(0.2)
        
        self.net1 = nn.Sequential(layers1)
        
        # net2: fusing net
        hidden_lst2 = [2 * hidden_lst1[-1]] + opts.D_hidden2 + [self.num_cls + 1]
        layers2 = OrderedDict()
        for n, (dim_in, dim_out) in enumerate(zip(hidden_lst2, hidden_lst2[1::])):
            layers2['fc%d' % n] = nn.Linear(dim_in, dim_out, bias = False)
            if n < len(hidden_lst2) - 2:
                layers2['bn%d' % n] = nn.BatchNorm1d(dim_out)
                layers2['leaky_relu%d' % n] = nn.LeakyReLU(0.2)
        
        layers2['logsoftmax'] = nn.LogSoftmax()
        self.net2 = nn.Sequential(layers2)
        
        
    def forward(self, feat_ref, feat_tar):
        bsz, feat_size = feat_ref.size()
        
        x = torch.cat((feat_ref, feat_tar), dim = 0)
        x = self.net1(x)
        
        y = torch.cat((x[0:bsz,:], x[bsz::,:]), dim = 1).contiguous()
        
        return self.net2(y)