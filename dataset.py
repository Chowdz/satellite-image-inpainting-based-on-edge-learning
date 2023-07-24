"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/7/14 17:26 
"""

import glob
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.feature import canny
from skimage.color import rgb2gray


class SatelliteDateset(Dataset):
    def __init__(self, image_root, mask_root, sigma=0.7, low_threshold=0.05, high_threshold=0.1):
        super(SatelliteDateset, self).__init__()

        self.image_files = sorted(glob.glob(image_root + '/*.*'))
        self.mask_files = sorted(glob.glob(mask_root + '/*.*'))

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)

        self.sigma = sigma
        self.l_thres = low_threshold
        self.h_thres = high_threshold

    def __getitem__(self, index):
        mask = Image.open(self.mask_files[index % len(self.mask_files)])
        mask_tensor = self.image_to_tensor()(mask)
        edge_tensor, gray_tensor, img_tensor = self.image_to_edge(index=index)

        return mask_tensor, edge_tensor, gray_tensor, img_tensor

    def __len__(self):
        return self.number_image

    def tensor_to_image(self):
        return transforms.ToPILImage()

    def image_to_tensor(self):
        return transforms.ToTensor()

    def image_to_edge(self, index):
        img = Image.open(self.image_files[index % len(self.image_files)])
        gray_image = rgb2gray(np.array(img))
        edge_tensor = self.image_to_tensor()(Image.fromarray(
            canny(gray_image, sigma=self.sigma, low_threshold=self.l_thres, high_threshold=self.h_thres,
                  mode='constant')))
        gray_tensor = self.image_to_tensor()(Image.fromarray(gray_image))
        img_tensor = self.image_to_tensor()(img)
        return edge_tensor, gray_tensor, img_tensor


if __name__ == '__main__':
    image_size = (256, 256)
    image_root = r'E:\PycharmProjects\DeepLearningStudy\Satellite_Image_Inpainting\Example\Image'
    mask_root = r'E:\PycharmProjects\DeepLearningStudy\Satellite_Image_Inpainting\Example\Mask'
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    dataset = SatelliteDateset(image_root, mask_root)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, img in enumerate(dataloader):
        print(i)
        print(img[0].shape, img[1].shape, img[2].shape, img[3].shape)
        break



