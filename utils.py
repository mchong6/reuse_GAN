import sys
import os
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
from torch.autograd import grad as torch_grad
import sobol_seq
import ghalton
from scipy.stats import norm
from fid import calculate_fid_given_paths

def check_nan(z):
    if torch.isnan(z).any():
        raise ValueError("z value is nan!")

def generate_dataset(opt, gen_model):
    z_sampler = randn_sampler(opt.sampler, opt.num_z)
    count = 0
    for i in range(math.ceil(1e4/opt.batch_size)):
        with torch.no_grad():
            z = z_sampler.next(opt.batch_size).cuda()
            _, fake_img = gen_model(z)
            fake_img = to_img(fake_img.cpu().data)
            
            for i in range(fake_img.size(0)):
                save_image(fake_img[i], '%s/%d.png'%(opt.gen_dir, count))
                count += 1
                if count >=10000:
                    break

def calculate_fid(opt):
    fid = calculate_fid_given_paths([opt.fid_path, opt.gen_dir])
    #clear gpu if needed
    torch.cuda.empty_cache()

    return fid

class randn_sampler():
    def __init__(self, sampler, ndim, skip=0, randomized=False):
        self.ndim = ndim
        self.randomized = randomized

        seed = 1 if sampler == 'halton' else 5 # first few sobol points are bad
        seed += skip

        if sampler == 'halton':
            self.sampler = halton_sampler(ndim, seed)
        elif sampler == 'sobol':
            self.sampler = sobol_sampler(ndim, seed)
        else:
            self.sampler = None

    def next(self, batch_size):
        if self.sampler is None:
            return torch.randn([batch_size, self.ndim])
        else:
            output = self.sampler.next(batch_size)
            if self.randomized:
                output = np.remainder(output + np.random.uniform(size=output.shape), 1)

        return torch.FloatTensor(norm.ppf(output))

class halton_sampler():
    def __init__(self, ndim, seed):
        self.gen = ghalton.GeneralizedHalton(ndim, seed)

    def next(self, batch_size):
        output = np.array(self.gen.get(batch_size))
        return output

class sobol_sampler():
    def __init__(self, ndim, seed):
        self.ndim = ndim
        self.gen = sobol_seq.i4_sobol_generate(ndim, seed)

    def next(self, batch_size):
        output = np.array(self.gen.next(self.ndim, batch_size))
        return output

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def generate_images(opt, gen_model, epoch, fixed_z=None):
    with torch.no_grad():
        z = torch.randn([opt.batch_size, opt.num_z]).cuda() if fixed_z is None else fixed_z
        _, fake_img = gen_model(z) 
        pic = to_img(fake_img.cpu().data)
        save_image(pic, '%s/fake_%d.png'%(opt.sample_dir, epoch))


def adjust_learning_rate(optimizers, epoch):        
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            if epoch == 30:
                new_lr = current_lr / 2
            elif epoch == 50:
                new_lr = current_lr / 5
            else:
                new_lr = current_lr
            param_group['lr'] = new_lr
            
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        elif self.type == 'wgan-gp':
            if is_real:
                outputs = -outputs
            return outputs.mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            if self.type == 'nsgan':
                outputs = torch.sigmoid(outputs)
            loss = self.criterion(outputs, labels)
            return loss

def gradient_penalty(real_data, generated_data, disc):
    batch_size = real_data.size(0)

    # Calculate interpolation
    alpha = torch.rand([batch_size]).cuda()
    
    # We don't know how many dimensions our input is
    num_dim = real_data.dim()
    for i in range(num_dim-1):
        alpha.unsqueeze_(-1)
    interpolated = (alpha * real_data + (1 - alpha) * generated_data).detach().requires_grad_()

    # Calculate probability of interpolated examples
    prob_interpolated = disc(interpolated)
    
    # sometimes we return features too
    if type(prob_interpolated) is list or type(prob_interpolated) is tuple:
        prob_interpolated = prob_interpolated[-1]

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 10 * ((gradients_norm - 1) ** 2).mean()

def permute_all(source, replacement=True):
    output = source.clone()
    for i in range(output.size(1)):
        sample = np.random.choice(output.size(0), output.size(0), replace=replacement)
        output[:, i] = output[sample, i]
    return output
