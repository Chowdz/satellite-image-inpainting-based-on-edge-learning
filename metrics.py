"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/7/17 14:59 
"""

import glob
import torch
from torchvision import transforms
from torchvision import models
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from pytorch_fid import fid_score
from paddle_msssim import ssim, ms_ssim
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio


class Metrics:

    def __init__(self, raw_tensor, gen_tensor):
        self.raw_tensor = raw_tensor
        self.gen_tensor = gen_tensor

    def psnr_value(self):
        return PeakSignalNoiseRatio().cuda()(self.raw_tensor, self.gen_tensor)

    def ssim_value(self):
        return StructuralSimilarityIndexMeasure().cuda()(self.raw_tensor, self.gen_tensor)

    def mape_value(self):
        return MeanAbsolutePercentageError().cuda()(self.raw_tensor, self.gen_tensor)


if __name__ == '__main__':
    t1 = torch.rand([10, 3, 256, 256]).cuda()
    t2 = torch.rand([10, 3, 256, 256]).cuda()
    metric = Metrics(t1, t2)

    psnr = metric.psnr_value()
    print(psnr)

    ssim = metric.ssim_value()
    print(ssim)

    mape = metric.mape_value()
    print(mape)
