import os
import torch
from torch import nn
import torch.nn.functional as F
# from torch.nn.utils import spectral_norm
import math
import utils
import functools

def get_activations(activations='relu'):
    if activations == 'relu':
        activation_layer = functools.partial(nn.ReLU, inplace=True)
    elif activations == 'leaky':
        activation_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    else:
        raise NotImplementedError('activations layer [%s] is not found' % activation_layer)
    return activation_layer

def get_dropout(mode):
    return functools.partial(nn.Dropout) if mode else functools.partial(Identity)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1

def init_weights(model, init_type='xavier', gain=0.02):
   '''
   initialize network's weights
   init_type: normal | xavier | kaiming | orthogonal
   '''

   def init_func(m):
       classname = m.__class__.__name__
       if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
           if init_type == 'normal':
               nn.init.normal_(m.weight.data, 0.0, gain)
           elif init_type == 'xavier':
#                nn.init.xavier_normal_(m.weight.data, gain=gain)
                nn.init.xavier_uniform_(m.weight.data)
           elif init_type == 'kaiming':
               nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
           elif init_type == 'orthogonal':
               nn.init.orthogonal_(m.weight.data, gain=gain)

           if hasattr(m, 'bias') and m.bias is not None:
               nn.init.constant_(m.bias.data, 0.0)

       elif classname.find('BatchNorm2d') != -1:
           nn.init.normal_(m.weight.data, 1.0, gain)
           nn.init.constant_(m.bias.data, 0.0)

   model.apply(init_func)
# def init_weights(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv2d:
#         xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()
        pass
    def forward(self, input):
        return input

class upblock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral):
        super(upblock, self).__init__()
        self.conv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), use_spectral),
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1), use_spectral),
#             PixelNorm(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        return self.conv1(x)
 
class lastconv(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral):
        super(lastconv, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), use_spectral),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.conv(x)

class generator(nn.Module):
    def __init__(self, opt):
        super(generator, self).__init__()
        self.num_skip = opt.num_skip
        conv_dim = 128
        
        num_up = int(math.log(opt.image_size, 2)) - 2 #start from 4x4
        in_dim = conv_dim*2**(num_up-1)
        out_dim = in_dim
        
        decoder = [spectral_norm(nn.ConvTranspose2d(opt.num_z, in_dim, 4, 1, 0), opt.use_spectral),
                  nn.BatchNorm2d(in_dim),
                  nn.ReLU(True),]
        # 64 -- 4
        # 128 -- 5
        for i in range(num_up):
            decoder.append(upblock(in_dim, out_dim, opt.use_spectral))
            in_dim = out_dim
            out_dim = out_dim // 2
            
        decoder.append(lastconv(in_dim, 3, opt.use_spectral))
        self.decoder = nn.ModuleList(decoder)
        
        
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        decode_len = len(self.decoder)-self.num_skip
        
        for i in range(decode_len):
            x = self.decoder[i](x)
        feat = x
        
        for i in range(self.num_skip):
            x = self.decoder[i+decode_len](x)
        
        return feat, x
    

class downblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_spectral, activations):
        super(downblock, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), use_spectral),
            activations(),
        )
        
    def forward(self, x):
        return self.conv(x)

class discriminator(nn.Module):
    # initializers
    def __init__(self, opt):
        super(discriminator, self).__init__()
        self.num_skip = opt.num_skip
        num_down = int(math.log(opt.image_size, 2)) - 2 # down to [4,4]
        in_dim = 3
        out_dim = 128
        activations = get_activations(opt.activations)
        
        layer  = []
        for i in range(num_down):
            layer.append(downblock(in_dim, out_dim, 4, 2, 1, use_spectral=opt.use_spectral, activations=activations))
            in_dim = out_dim
            out_dim *= 2
            
        layer += [
                Flatten(), 
                spectral_norm(nn.Linear(in_dim*4*4, 1), opt.use_spectral),
            ]
        
        self.layer = nn.ModuleList(layer)
        
        
    # forward method
    def forward(self, x):
        for i in range(len(self.layer)):
            x = self.layer[i](x)
            
        return x,x,x
    
class LocalDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        self.im_size = opt.feat_size[-1]
        self.layer = nn.Sequential(
            nn.Conv2d(opt.feat_size[1]+128, 512, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, y, M):
        # y [B, 128]
        # M [B, 3, 64, 64]
        y = y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.im_size, self.im_size)
        y = torch.cat((M, y), dim=1) 

        return self.layer(y)

class GlobalDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        im_size = opt.feat_size[-1]
        num_down = int(math.log(im_size, 2)) - 3 # final conv to be 8x8
        in_dim = opt.feat_size[1]
        out_dim = 128
        
        layer = []
        for i in range(num_down):
            layer += [
                nn.Conv2d(in_dim, out_dim, 3,2,1),
                nn.LayerNorm([out_dim, im_size//(2**(i+1)), im_size//(2**(i+1))]),
                nn.LeakyReLU(0.2, True),
            ]
            in_dim = out_dim
            out_dim *= 2
            
        layer.append(Flatten())
        self.conv1 = nn.Sequential(*layer)
        
        self.conv2 = nn.Sequential(
            nn.Linear(in_dim*8*8 +128, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1)
        )

    def forward(self, y, M):
        h = self.conv1(M)
        h = torch.cat((y, h), dim=1)
        return self.conv2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()        
        self.layer = nn.Sequential(
            nn.Linear(128, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 200),
            nn.LeakyReLU(0.2, True),
            nn.Linear(200, 1),
        )

    def forward(self, y):
        return self.layer(y)

class MI(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.prior_type = opt.prior_type
        self.global_d = GlobalDiscriminator(opt)
        self.local_d = LocalDiscriminator(opt)
        
        # Generator don't have prior loss because z is normal
        if opt.use_MI is not "gen":
            self.prior_d = PriorDiscriminator()
            self.adv_criterion = utils.AdversarialLoss(opt.loss)

    def forward(self, feat, enc, is_disc=True):
        feat_prime = torch.cat((feat[1:], feat[0, None]), dim=0)
#         feat_prime = feat[torch.randperm(feat.size(0))]
        local_loss = self.calculate_MI(feat, feat_prime, enc, self.global_d)
        global_loss = self.calculate_MI(feat, feat_prime, enc, self.local_d)
        
        if self.opt.use_MI == 'gen':
            return local_loss, global_loss
        else:
            if is_disc:
                if self.prior_type == 'uniform':
                    prior = torch.rand_like(enc).cuda()
                else:
                    prior = torch.randn_like(enc).cuda()
                prior_loss = self.adv_criterion(self.prior_d(enc), False, is_disc) + \
                self.adv_criterion(self.prior_d(prior), True, is_disc)
                if self.opt.loss == 'wgan-gp':
                    err_penalty = utils.gradient_penalty(enc, prior, self.prior_d)
                    prior_loss += err_penalty

            else:
#                 prior_loss = self.adv_criterion(self.prior_d(enc), False, is_disc) + \
#                 self.adv_criterion(self.prior_d(prior), True, is_disc)
                prior_loss = self.adv_criterion(self.prior_d(enc), True, is_disc)

            return local_loss, global_loss, prior_loss

    def calculate_MI(self, feat, feat_prime, enc, model):
        Ej = -F.softplus(-model(enc, feat)).mean()
        Em = F.softplus(model(enc, feat_prime)).mean()
        loss = (Em - Ej)
        return loss
