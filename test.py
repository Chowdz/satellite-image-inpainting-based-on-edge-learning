"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/7/17 21:30 
"""


import os
import csv
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
from fid_score import calculate_fid_given_paths
from networks import EdgeGenerator, InpaintGenerator

opt = Options().parse

cuda = torch.cuda.is_available()

pic_shape = (opt.HEIGHT, opt.WIDTH)


# Initialize Generator
edge_generator = EdgeGenerator()
inpaint_generator = InpaintGenerator()

# Load Generator
edge_generator.load_state_dict(torch.load(r'../Save_models/edge_generator_' + str(opt.EPOCH) + '.pth'))
inpaint_generator.load_state_dict(torch.load(r'../Save_models/inpaint_generator_' + str(opt.EPOCH) + '.pth'))


# Set GPU
if cuda:
    edge_generator = edge_generator.cuda()
    inpaint_generator = inpaint_generator.cuda()



Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Set DataLoader
dataset_test = SatelliteDateset(opt.TEST_IMAGE_ROOT, opt.TEST_MASK_ROOT)
dataloader_test = DataLoader(dataset_test, batch_size=opt.TEST_BATCH_SIZE, shuffle=True, num_workers=0)


# --------------------- #
#    Testing Process   #
# --------------------- #

with open('../Qualitative_Comparison_Result_Scanline.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PSNR', 'SSIM', 'MAPE'])

    with torch.no_grad():
        edge_generator.eval()
        inpaint_generator.eval()
        psnr_, ssim_, mape_ = 0, 0, 0
        for index, imgs in enumerate(dataloader_test):

            mask, edge, img = imgs[0].type(Tensor), imgs[1].type(Tensor), imgs[3].type(Tensor)
            edge_miss, img_miss = 1 - edge * (1 - mask), img * (1 - mask) + mask

            edge_generator_input = Variable(torch.cat([mask, edge_miss], dim=1), requires_grad=False)
            edge_gen = edge_generator(edge_generator_input)

            inpaint_gen_input = Variable(torch.cat([img_miss, edge_gen], dim=1), requires_grad=False)
            inpaint_gen = inpaint_generator(inpaint_gen_input)

            inpaint_gen_split = torch.split(inpaint_gen, split_size_or_sections=1)
            for i, pic in enumerate(inpaint_gen_split):
                output_pic = dataset_test.tensor_to_image()(pic.reshape([-1, opt.HEIGHT, opt.WIDTH]))
                output_pic.save(opt.OUTPUT_ROOT + str(index) + '_' + str(i) + '.TIF')

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
            comp_pic = dataset_test.tensor_to_image()(comp)
            comp_pic.save(opt.TEST_RESULT_ROOT + opt.TEST_MASK_ROOT[13: -2] + str(index) + '.TIF')

            metric = Metrics(img, inpaint_gen)
            psnr = metric.psnr_value()
            ssim = metric.ssim_value()
            mape = metric.mape_value()

            writer.writerow([psnr.item(), ssim.item(), mape.item()])

            sys.stdout.write(
                "\033[36m[Batch %d/%d] \033[0m \033[32m[PSNR: %f] [SSIM: %f] [MAPE: %f]\033[0m\n"
                % (index, len(dataloader_test), psnr, ssim, mape))

            psnr_ += psnr
            ssim_ += ssim
            mape_ += mape

    file.close()

    psnr_average = psnr_ / len(dataloader_test)
    ssim_average = ssim_ / len(dataloader_test)
    mape_average = mape_ / len(dataloader_test)

fid_sum = 0
for _ in range(10):
    fid_score = calculate_fid_given_paths(opt.TEST_IMAGE_ROOT, opt.OUTPUT_ROOT)
    fid_sum += fid_score
fid = fid_sum / 10
print(fid)

# print(f'PSNR: {psnr_average}, SSIM: {ssim_average}, MAPE: {mape_average}')




