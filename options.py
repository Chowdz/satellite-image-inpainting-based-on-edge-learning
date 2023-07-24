"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/7/17 21:32 
"""


import argparse
import pprint
from collections import OrderedDict


class Options:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--TRAIN_IMAGE_ROOT", type=str, default='../Dataset_crop/train/')
        parser.add_argument("--TRAIN_MASK_ROOT", type=str, default='../Mask/train/')
        parser.add_argument("--TEST_IMAGE_ROOT", type=str, default='../Test_Data/TestImage/')
        parser.add_argument("--TEST_MASK_ROOT", type=str, default='../Test_Data/ScanlineMask/')
        parser.add_argument("--OUTPUT_ROOT", type=str, default='../Test_Data/InpaintImage/')
        parser.add_argument("--TEST_RESULT_ROOT", type=str, default='../Test_Data/TestResult/')
        parser.add_argument("--EPOCH", type=int, default=10, help="epoch to start training from")
        parser.add_argument("--N_EPOCH", type=int, default=202, help="number of epochs of training")
        parser.add_argument("--BATCH_SIZE", type=int, default=2, help="size of the batches")
        parser.add_argument("--TEST_BATCH_SIZE", type=int, default=10, help="size of the test batches")
        parser.add_argument("--LR", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--D2G_LR", type=float, default=0.1, help="discriminator/generator learning rate ratio")
        parser.add_argument("--B1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--B2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--STRUCTURE_LOSS_WEIGHT", type=float, default=5,
                            help="edge generator structure loss weight")
        parser.add_argument("--FM_LOSS_WEIGHT", type=float, default=20, help="feature-matching loss weight")
        parser.add_argument("--EDGE_ADV1_LOSS_WEIGHT", type=float, default=1, help="adversarial loss weight")
        parser.add_argument("--L1_LOSS_WEIGHT", type=float, default=0.1, help="l1 loss weight")
        parser.add_argument("--INPAINT_ADV2_LOSS_WEIGHT", type=float, default=1, help="adversarial loss weight")
        parser.add_argument("--STYLE_LOSS_WEIGHT", type=float, default=250, help="style loss weight")
        parser.add_argument("--CONTENT_LOSS_WEIGHT", type=float, default=0.1, help="perceptual loss weight")
        parser.add_argument("--HEIGHT", type=int, default=256, help="high res. image height")
        parser.add_argument("--WIDTH", type=int, default=256, help="high res. image width")
        parser.add_argument("--CHANNELS", type=int, default=3, help="number of image channels")
        parser.add_argument("--SAMPLE_INTERVAL", type=int, default=100, help="interval between saving image samples")
        parser.add_argument("--CHECKPOINT_INTERVAL", type=int, default=1, help="interval between model checkpoints")
        self.opt = parser.parse_args()

    @property
    def parse(self):
        opts_dict = OrderedDict(vars(self.opt))
        pprint.pprint(opts_dict)

        return self.opt