from __future__ import division
from __future__ import print_function

import torch
import torchvision
from models import *
from swin_transformer import *


import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tqdm import tqdm
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F

from models import *
from utils import *
from einops import rearrange, repeat

import matplotlib.pyplot as plt

beta1 = 0
beta2 = 0.99
lr_gen = 0.0001
lr_dis = 0.00001


def train(batch, encoder, decoder, loss, optim_enc, optim_dec, optim_dis):
    features = encoder.forward(batch)
    image = decoder(features)

    real_valid = discriminator(batch)
    fake_imgs = image.detach()

    fake_valid = discriminator(fake_imgs)
    optim_dis.zero_grad()
    loss_dis = torch.mean(nn.ReLU(inplace=True)(1.0 - real_valid)).cuda() + torch.mean(
        nn.ReLU(inplace=True)(1 + fake_valid)).cuda()

    loss_dis.backward()
    optim_dis.step()

    optim_enc.zero_grad()
    optim_dec.zero_grad()
    fake_valid = discriminator(decoder(encoder(batch)))
    gener_loss = -torch.mean(fake_valid).cuda()
    gener_loss.backward()

    optim_enc.step()
    optim_dec.step()

    return image, loss_dis, gener_loss

if __name__ == '__main__':
    img_size = 32
    loss = nn.MSELoss()
    transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    decoder = Generator(depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4,
                          drop_rate=0.5)  # ,device = device)

    decoder = decoder.cuda()

    discriminator = Discriminator(diff_aug="translation,cutout,color", image_size=32, patch_size=4, input_channel=3, num_classes=1,
                                  dim=384, depth=7, heads=4, mlp_ratio=4,
                                  drop_rate=0.)

    discriminator = discriminator.cuda()


    optim_dec = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=lr_gen,
                           betas=(beta1, beta2))

    optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),lr=lr_dis, betas=(beta1, beta2))


    encoder = SwinTransformer(img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False
    )

    optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=lr_gen,
                           betas=(beta1, beta2))

    encoder = encoder.cuda()

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU()
    for i in range(10):
        for index, (img, target) in enumerate(train_loader):

            guess = encoder.forward(img.cuda())
            guess = relu(guess)
            loss = criterion(softmax(guess), target.cuda())
            loss.backward()
            optim_enc.step()

            if index % 10 == 1:
                print(loss)



