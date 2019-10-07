import os
import json
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import shutil
import models
from dataloader import im_dataset
import utils
import argparse
from collections import OrderedDict
import torchvision
from torchvision import transforms
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64, help='input image size')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--num_z', type=int, default=128, help='latent variable dimension. 64 for celeba, 8 for mnist')
parser.add_argument('--num_skip', type=int, default=0, help='which intermediate features to use for MI. 0 is input/output pixels')
parser.add_argument('--beta1', type=float, default=0, help='Weight of mutual info term')
parser.add_argument('--beta2', type=float, default=0.9, help='Weight of mutual info term')
parser.add_argument('--loss', default='hinge', help='hinge|lsgan|nsgan')
parser.add_argument('--out_dir', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--restore_gen', default='', type=str)
parser.add_argument('--restore_disc', default='', type=str)
parser.add_argument('--use_spectral', type=int, default=1, help='frequency of printing progress')
parser.add_argument('--sampler', type=str, default='normal', help='normal|sobol|halton')
parser.add_argument('--sampler_mode', type=str, default='normal', help='normal(just run the sequence)|fixed(restart sequence every epoch)|fixed-random(restart every epoch with random seed)')
parser.add_argument('--randomized', type=int, default=0, help='randomized QMC')
parser.add_argument('--use_dropout', type=int, default=0, help='frequency of printing progress')
parser.add_argument('--activations', type=str, default='leaky', help='frequency of printing progress')
parser.add_argument('--prior_type', type=str, default='uniform', help='uniform | normal')
parser.add_argument('--reuse', type=str, default='disc', help='gen | disc')
parser.add_argument('--warm_up_epoch', type=int, default=0, help='frequency of printing progress')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of printing progress')
parser.add_argument('--save_freq', type=int, default=1, help='frequency of saving models and generating images')
parser.add_argument("--gpu", default="", type=str, help='GPU to use (leave blank for CPU only)')
#parser.add_argument('--fid_path', type=str, required=True, help='path to precalculated fid npz')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

opt.out_dir = os.path.join('./output', str(opt.image_size), opt.out_dir)
opt.gen_dir = os.path.join(opt.out_dir, 'gen')
opt.sample_dir = os.path.join(opt.out_dir, 'samples')
os.makedirs(opt.out_dir, exist_ok=True)
os.makedirs(opt.gen_dir, exist_ok=True)
os.makedirs(opt.sample_dir, exist_ok=True)
    
# save argparse to textfile
with open('%s/argparse.txt'%opt.out_dir, 'w') as f:
    json.dump(opt.__dict__, f, indent=4)

# textfile for fid
best_fid = 1e5
fid_file = open(os.path.join(opt.out_dir, 'fid.txt'),"w") 
fid_list = []
    
opt.fid_path = './pytorch_celeba_full.npz' if opt.image_size == 64 else './pytorch_celeba_full128.npz'

transform_train = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = torchvision.datasets.CelebA(root='../MI_GAN/data', download=False, transform=transform_train)
# dataset = im_dataset('../MI_disentangle/data/img_align_celeba', opt.image_size)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

gen_model = models.generator(opt).cuda().train()
disc_model = models.discriminator(opt).cuda().train()
models.init_weights(gen_model)
models.init_weights(disc_model)

if opt.restore_gen != '':
    gen_model.load_state_dict(torch.load('./output/%d/%s/gen.pth'%(opt.image_size, opt.restore_gen)))
if opt.restore_disc != '':
    disc_model.load_state_dict(torch.load('./output/%d/%s/disc.pth'%(opt.image_size, opt.restore_disc)))


assert opt.sampler in ('normal', 'sobol', 'halton'), 'Unknown sampler parameter'
assert opt.sampler_mode in ('normal', 'fixed', 'fixed-random'), 'Unknown sampler_mode parameter'

z_sampler= utils.randn_sampler(opt.sampler, opt.num_z, randomized=opt.randomized)
adv_criterion = utils.AdversarialLoss(opt.loss).cuda()

gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=1e-4, betas=(0, 0.9))
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=4e-4, betas=(0, 0.9))
fixed_z = torch.randn([opt.batch_size, opt.num_z]).cuda()

def update_disc(img, z, all_losses):
    g_feat, fake_img = gen_model(z)

    d_real_feat, d_real_enc, real_out = disc_model(img)
    d_fake_feat, d_fake_enc, fake_out = disc_model(fake_img.detach())

    err_real = adv_criterion(real_out, True, True)
    err_fake = adv_criterion(fake_out, False, True)
    
    d_loss = err_real + err_fake
    if opt.loss == 'wgan-gp':
        err_penalty = utils.gradient_penalty(img, fake_img.detach(), disc_model)
        d_loss += err_penalty
    
    all_losses['d_real'] = round(err_real.item(),4 )
    all_losses['d_fake'] = round(err_fake.item(), 4)
    
    disc_optimizer.zero_grad()
    d_loss.backward()
    disc_optimizer.step()

def update_gen(img, z, all_losses):
    g_feat, fake_img = gen_model(z)
    
    _, _, fake_out = disc_model(fake_img)
    g_loss = adv_criterion(fake_out, True, False)
    all_losses['g_loss'] = round(g_loss.item(), 4)
    
    gen_optimizer.zero_grad()
    g_loss.backward()
    gen_optimizer.step()

for epoch in range(opt.num_epochs):
    for batch_idx, (img, att) in enumerate(dataloader):
        start_time = time.time()
        all_losses = OrderedDict() 
        img = img.cuda()

        z = z_sampler.next(img.size(0)).cuda()
        utils.check_nan(z)
        if epoch < opt.warm_up_epoch:
            z2 = z_sampler.next(img.size(0)).cuda()
        else:
            z2 =  z

        if opt.reuse == 'gen':
            update_disc(img, z, all_losses)
            update_gen(img, z2, all_losses)
        else:
            update_gen(img, z, all_losses)
            update_disc(img, z2, all_losses)
        
        end_time = time.time()
        if batch_idx % opt.print_freq == 0:
            print('epoch [{}/{}] batch [{}/{}] Time {:.3f}'.format(epoch, opt.num_epochs, batch_idx, len(dataloader), end_time-start_time), ', '.join(['{0}: {1}'.format(k, v) for k,v in all_losses.items()]))

            utils.generate_images(opt, gen_model, epoch, fixed_z)


    if epoch % opt.save_freq == 0:
        utils.generate_images(opt, gen_model, epoch, fixed_z)
        # generate dataset and calculate dataset
        utils.generate_dataset(opt, gen_model)
        fid = utils.calculate_fid(opt)
        fid_list.append(fid)
        fid_file.write(f'Epoch {epoch}: {fid} \n')

        torch.save(gen_model.state_dict(), '%s/gen.pth'%(opt.out_dir))
        torch.save(disc_model.state_dict(), '%s/disc.pth'%(opt.out_dir))

        print(f'FID: {fid}, Best FID: {best_fid}')
        if fid <= best_fid:
            best_fid = fid
            shutil.copyfile('%s/gen.pth'%(opt.out_dir),'%s/best_gen.pth'%(opt.out_dir))
            shutil.copyfile('%s/disc.pth'%(opt.out_dir),'%s/best_disc.pth'%(opt.out_dir))


fid_file.close() 
plt.figure()
plt.plot(fid_list)
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.savefig(os.path.join(opt.out_dir, 'FID.png'))
