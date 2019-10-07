import os
import json
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import math

import models
from dataloader import im_dataset
import utils
import argparse
from torchvision.utils import save_image
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64, help='input image size')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_z', type=int, default=128, help='latent variable dimension. 64 for celeba, 8 for mnist')
parser.add_argument('--num_skip', type=int, default=0, help='latent variable dimension. 64 for celeba, 8 for mnist')
parser.add_argument('--sigma', type=float, default=1.0, help='latent variable dimension. 64 for celeba, 8 for mnist')
parser.add_argument('--out_dir', default='temp')
parser.add_argument('--restore_gen', default='', type=str)
parser.add_argument('--use_spectral', type=int, default=1, help='frequency of printing progress')
parser.add_argument("--gpu", default="", type=str, help='GPU to use (leave blank for CPU only)')
parser.add_argument('--sampler', type=str, default='normal', help='normal|sobol|halton')
parser.add_argument('--skip', type=int, default=0, help='frequency of printing progress')


opt = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

opt.out_dir = os.path.join('./output/generated', opt.out_dir)
os.makedirs(opt.out_dir, exist_ok=True)

assert opt.sampler in ('normal', 'sobol', 'halton'), 'Unknown sampler parameter'

gen_model = models.generator(opt).cuda().train()
try:
    gen_model.load_state_dict(torch.load('./output/%d/%s/best_gen.pth'%(opt.image_size, opt.restore_gen)))
except:
    print("Best file not found, loading latest file")
    gen_model.load_state_dict(torch.load('./output/%d/%s/gen.pth'%(opt.image_size, opt.restore_gen)))

z_sampler = utils.randn_sampler(opt.sampler, opt.num_z, skip=opt.skip)
    
count = 0

for i in tqdm(range(math.ceil(1e4/opt.batch_size))):
    with torch.no_grad():
        z = z_sampler.next(opt.batch_size).cuda()
        _, fake_img = gen_model(z)
        fake_img = utils.to_img(fake_img.cpu().data)
        
        for i in range(fake_img.size(0)):
            save_image(fake_img[i], '%s/%d.png'%(opt.out_dir, count))
            count += 1
            if count >=10000:
                break
