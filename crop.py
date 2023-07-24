"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/7/8 15:51 
"""

import os
from PIL import Image
from torchvision import transforms


class CropImage:
    def rotation(self, path, degree=10, crop=5120):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=(degree, degree)),
                                    transforms.CenterCrop(crop),
                                    transforms.ToPILImage()])
        return trans(Image.open(path))

    def cut(self, img, cutnumber=20):
        width = img.size[0]
        height = img.size[1]
        for i in range(cutnumber):
            for j in range(cutnumber):
                cut_pic = img.crop([width / cutnumber * i, height / cutnumber * j, width / cutnumber * (i + 1),
                                   height / cutnumber * (j + 1)])
                yield cut_pic


if __name__ == '__main__':
    cropimage = CropImage()
    input_path = r'../Test_Data/Dataset_off_truecolor'
    output_path = r'../Test_Data/Dataset_off_crop'
    input_li = os.listdir(input_path)
    for ind, file in enumerate(input_li):
        pic_path = input_path + '/' + file
        img = cropimage.rotation(pic_path)
        for index, cut_img in enumerate(cropimage.cut(img)):
            cut_img.save(output_path + '/' + file.replace('.TIF', '') + '_' + str(index) + '.TIF')
        print(f'cut complete: {ind + 1}')