"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/7/14 16:45 
"""

import os
import sys
import time
import datetime
import torch
import argparse
import torch.nn as nn
import numpy as np
from metrics import Metrics
from dataset import SatelliteDateset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from options import Options
from loss import PerceptualLoss, StyleLoss
from networks import EdgeGenerator, InpaintGenerator, Discriminator

opt = Options().parse


cuda = torch.cuda.is_available()

pic_shape = (opt.HEIGHT, opt.WIDTH)

# Initialize generator and discriminator
edge_generator = EdgeGenerator()
edge_discriminator = Discriminator(in_channels=1)
inpaint_generator = InpaintGenerator()
inpaint_discriminator = Discriminator(in_channels=3)

# Set LOSS
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
perceptual_loss = PerceptualLoss()
style_loss = StyleLoss()

# Set GPU
if cuda:
    edge_generator = edge_generator.cuda()
    edge_discriminator = edge_discriminator.cuda()
    inpaint_generator = inpaint_generator.cuda()
    inpaint_discriminator = inpaint_discriminator.cuda()
    l1_loss = l1_loss.cuda()
    l2_loss = l2_loss.cuda()
    perceptual_loss = perceptual_loss.cuda()
    style_loss = style_loss.cuda()

# Set Checkpoints
if opt.EPOCH != 0:
    edge_generator.load_state_dict(
        torch.load(r'../Save_models/edge_generator_' + str(opt.EPOCH) + '.pth'))
    edge_discriminator.load_state_dict(
        torch.load(r'../Save_models/edge_discriminator_' + str(opt.EPOCH) + '.pth'))
    inpaint_generator.load_state_dict(
        torch.load(r'../Save_models/inpaint_generator_' + str(opt.EPOCH) + '.pth'))
    inpaint_discriminator.load_state_dict(
        torch.load(r'../Save_models/inpaint_discriminator_' + str(opt.EPOCH) + '.pth'))

# Set Optimizers
optimizer_G1 = torch.optim.Adam(edge_generator.parameters(), lr=opt.LR, betas=(opt.B1, opt.B2))
optimizer_D1 = torch.optim.Adam(edge_discriminator.parameters(), lr=opt.LR * opt.D2G_LR, betas=(opt.B1, opt.B2))
optimizer_G2 = torch.optim.Adam(inpaint_generator.parameters(), lr=opt.LR, betas=(opt.B1, opt.B2))
optimizer_D2 = torch.optim.Adam(inpaint_discriminator.parameters(), lr=opt.LR * opt.D2G_LR * 0.1,
                                betas=(opt.B1, opt.B2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Set DataLoader
dataset_train = SatelliteDateset(opt.TRAIN_IMAGE_ROOT, opt.TRAIN_MASK_ROOT)
dataloader_train = DataLoader(dataset_train, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=0)



# --------------------- #
#    Training Process   #
# --------------------- #
prev_time = time.time()
inpaint_gen_li, img_raw_li = [], []
for epoch in range(opt.EPOCH, opt.N_EPOCH):
    for index, imgs in enumerate(dataloader_train):

        # Determine the Edge Generator's input
        mask, edge, gray, img = imgs[0].type(Tensor), imgs[1].type(Tensor), imgs[2].type(Tensor), imgs[3].type(Tensor)
        edge_miss, gray_miss, img_miss = 1 - edge * (1 - mask), gray * (1 - mask) + mask, img * (1 - mask) + mask

        edge_generator_input = Variable(torch.cat([mask, edge_miss], dim=1), requires_grad=True)

        # Define the Reference Probability for Discriminators
        valid = Variable(torch.ones([opt.BATCH_SIZE, 1, 30, 30]).type(Tensor), requires_grad=False)
        fake = Variable(torch.zeros([opt.BATCH_SIZE, 1, 30, 30]).type(Tensor), requires_grad=False)

        # ----------------------------- #
        #    Training Edge Generators   #
        # ----------------------------- #
        optimizer_G1.zero_grad()

        # Generate a edge from mask, edge_miss, gray_miss
        edge_gen = edge_generator(edge_generator_input)

        # Merge the Gray and generated edge for put in the Discriminator
        edge_dis_real_input = Variable((1 - edge) * mask, requires_grad=True)
        edge_dis_fake_input = edge_gen * mask

        # Define the Edge Generator Loss
        edge_dis_real, conv_real = edge_discriminator(edge_dis_real_input)
        edge_dis_fake, conv_fake = edge_discriminator(edge_dis_fake_input)
        edge_gen_loss = l2_loss(edge_dis_fake, valid)


        # Define the Feature-Matching Loss
        edge_gen_fm_loss = 0
        for i in range(len(conv_real)):
            edge_gen_fm_loss += l2_loss(conv_fake[i], conv_real[i].detach())


        # My innovation: Define the Edge Structure Loss
        edge_structure_loss = Metrics(edge_dis_fake_input, edge_dis_real_input)
        edge_gen_structure_loss = - torch.log(edge_structure_loss.ssim_value())


        # Total Loss
        loss_G1 = opt.STRUCTURE_LOSS_WEIGHT * edge_gen_structure_loss \
                  + opt.FM_LOSS_WEIGHT * edge_gen_fm_loss \
                  + opt.EDGE_ADV1_LOSS_WEIGHT * edge_gen_loss
        print(f'edge_gen_loss: {opt.EDGE_ADV1_LOSS_WEIGHT * edge_gen_loss} edge_gen_fm_loss: {opt.FM_LOSS_WEIGHT * edge_gen_fm_loss} edge_gen_structure_loss: {opt.STRUCTURE_LOSS_WEIGHT * edge_gen_structure_loss}')



        loss_G1.backward()
        optimizer_G1.step()

        # --------------------------------- #
        #    Training Edge Discriminators   #
        # --------------------------------- #

        # Define the discriminate loss
        optimizer_D1.zero_grad()
        dis_real_loss1 = l2_loss(edge_dis_real, valid)
        dis_fake_loss1 = l2_loss(edge_dis_fake.detach(), fake)
        print(f'edge_dis_realloss: {dis_real_loss1}  edge_dis_fakeloss: {dis_fake_loss1}')


        loss_D1 = (dis_real_loss1 + dis_fake_loss1) / 2

        loss_D1.backward()
        optimizer_D1.step()

        # ----------------------------------- #
        #    Training Inpainting Generators   #
        # ----------------------------------- #

        # Determine the Inpainting Generator's input
        inpaint_generator_input = Variable(torch.cat([img_miss, edge_gen], dim=1), requires_grad=True)
        img_raw = Variable(img, requires_grad=True)

        optimizer_G2.zero_grad()

        # Generate the inpainted image from the generator
        inpaint_gen = inpaint_generator(inpaint_generator_input)

        # Define the adversarial loss
        inpaint_dis_fake, _ = inpaint_discriminator(inpaint_gen)

        inpaint_gen_loss = l2_loss(inpaint_dis_fake, valid)

        # Define the perceptual loss
        inpaint_perceptual_loss = perceptual_loss(inpaint_gen, img_raw)

        # Define the style loss
        inpaint_style_loss = style_loss(inpaint_gen, img_raw)

        # Define the L1 loss
        inpaint_l1_loss = l1_loss(inpaint_gen, img_raw) / torch.mean(mask)

        # Total loss
        loss_G2 = opt.INPAINT_ADV2_LOSS_WEIGHT * inpaint_gen_loss \
                  + opt.CONTENT_LOSS_WEIGHT * inpaint_perceptual_loss \
                  + opt.STYLE_LOSS_WEIGHT * inpaint_style_loss \
                  + opt.L1_LOSS_WEIGHT * inpaint_l1_loss
        print(f'inp_gen_loss: {opt.INPAINT_ADV2_LOSS_WEIGHT * inpaint_gen_loss} inp_per_loss: {opt.CONTENT_LOSS_WEIGHT * inpaint_perceptual_loss}')
        print(f'inp_sty_loss: {opt.STYLE_LOSS_WEIGHT * inpaint_style_loss} inp_l1_loss: {opt.L1_LOSS_WEIGHT * inpaint_l1_loss}')

        loss_G2.backward()
        optimizer_G2.step()

        # Compute training PSNR, SSIM, MAPE
        metric = Metrics(img_raw, inpaint_gen)
        psnr = metric.psnr_value()
        ssim = metric.ssim_value()
        mape = metric.mape_value()

        # --------------------------------------- #
        #    Training Inpainting Discriminators   #
        # --------------------------------------- #

        optimizer_D2.zero_grad()

        inpaint_dis_real, _ = inpaint_discriminator(img)
        dis_real_loss2 = l2_loss(inpaint_dis_real, valid)
        dis_fake_loss2 = l2_loss(inpaint_dis_fake.detach(), fake)
        print(f'dis_real_loss2: {dis_real_loss2} dis_fake_loss2: {dis_fake_loss2}')

        loss_D2 = (dis_real_loss2 + dis_fake_loss2) / 2
        loss_D2.backward()
        optimizer_D2.step()

        # Every n batches, save the inpainted image
        batches_done = epoch * len(dataloader_train) + index + 1
        if batches_done % opt.SAMPLE_INTERVAL == 0:

            real = torch.split(img, split_size_or_sections=1)
            real_edge = torch.split(1 - edge, split_size_or_sections=1)
            real_miss = torch.split(img_miss, split_size_or_sections=1)
            fake_edge_sole = torch.split(edge_gen, split_size_or_sections=1)
            fake_pic = torch.split(inpaint_gen, split_size_or_sections=1)

            real_comp = torch.cat(real, dim=3).reshape(opt.CHANNELS, opt.HEIGHT, -1)
            real_edge_comp = torch.cat(real_edge, dim=3).reshape(1, opt.HEIGHT, -1).expand(opt.CHANNELS, opt.HEIGHT, -1)
            real_miss_comp = torch.cat(real_miss, dim=3).reshape(opt.CHANNELS, opt.HEIGHT, -1)
            fake_edge_sole_comp = torch.cat(fake_edge_sole, dim=3).reshape(1, opt.HEIGHT, -1).expand(opt.CHANNELS,
                                                                                                     opt.HEIGHT, -1)
            fake_comp = torch.cat(fake_pic, dim=3).reshape(opt.CHANNELS, opt.HEIGHT, -1)

            comp = torch.cat([real_comp, real_edge_comp, real_miss_comp, fake_edge_sole_comp, fake_comp], dim=1)
            comp_pic = dataset_train.tensor_to_image()(comp)
            comp_pic.save(r'../Images/' + str(batches_done) + '.png')


        # ----------------- #
        #    Log Progress   #
        # ----------------- #

        # print the Loss by time
        batches_left = opt.N_EPOCH * len(dataloader_train) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\033[36m[Epoch %d/%d] [Batch %d/%d] [G1 loss: %f] [D1 loss: %f] [G2 loss: %f] [D2 loss: %f]\033[0m \033[32m[PSNR: %f] [SSIM: %f] [MAPE: %f]\033[0m\033[33mETA: %s\033[0m \n"
            % (epoch,opt.N_EPOCH,
               index, len(dataloader_train),
               loss_G1.item(), loss_D1.item(), loss_G2.item(),loss_D2.item(),
               psnr, ssim, mape,
               time_left))

    # Set the checkpoint
    if opt.CHECKPOINT_INTERVAL != -1 and epoch % opt.CHECKPOINT_INTERVAL == 0:
        torch.save(edge_generator.state_dict(), '../Save_models/edge_generator_' + str(epoch) + '.pth')
        torch.save(edge_discriminator.state_dict(), '../Save_models/edge_discriminator_' + str(epoch) + '.pth')
        torch.save(inpaint_generator.state_dict(), '../Save_models/inpaint_generator_' + str(epoch) + '.pth')
        torch.save(inpaint_discriminator.state_dict(), '../Save_models/inpaint_discriminator_' + str(epoch) + '.pth')
