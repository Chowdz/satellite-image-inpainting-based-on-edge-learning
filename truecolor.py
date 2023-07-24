"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : Admin
@Time : 2023/7/5 22:12 
"""



import numpy as np
from osgeo import gdal
import os


class ColorCombination:
    def read_image(self, filename):
        dataset = gdal.Open(filename)  ##调用Open函数打开遥感影像，并创建一个数据集：dataset
        width = dataset.RasterXSize  ##图像的列数
        height = dataset.RasterYSize  ##图像的行数
        image_proj = dataset.GetProjection()  ##获取投影信息
        image_geotrans = dataset.GetGeoTransform()  ##获取地理转换参数
        image_band = dataset.RasterCount  ##获取波段数
        image_data = dataset.ReadAsArray(0, 0, width, height)  ##将栅格数据读作能够用numpy操作的array
        del dataset  ##在获取完数据之后记得删除数据集，释放内存
        return width, height, image_proj, image_geotrans, image_band, image_data

    def write_image(self, filename, proj, trans, image_data):
        datatype = gdal.GDT_Byte ##定义数据的类型，我的是“uint16”，“uint8”对应的是“gdal.GDT_Byte”
        im_bands, im_width, im_height = image_data.shape  # 获取数据的波段数、宽、高
        driver = gdal.GetDriverByName('GTiff')  # 注册一个类型为“tif"的驱动
        dataset = driver.Create(filename, im_height, im_width, im_bands, datatype)  # 借助驱动来创建一个"tif"的数据集
        dataset.SetGeoTransform(trans)  # 将转换参数写到数据集里
        dataset.SetProjection(proj)  # 将投影信息写到数据集里
        for band in range(im_bands):  # range函数从0开始，例如3的话，遍历就是0，1，2
            dataset.GetRasterBand(band + 1).WriteArray(image_data[band])  # 遍历将三个波段的数据写入数据集
        del dataset



if __name__ == "__main__":
    run_merge = ColorCombination()
    input_dir = '../Test_Data/Dataset_off_B1B2B3'
    output_dir = '../Test_Data/Dataset_off_truecolor'
    father_list = os.listdir(input_dir)
    for index, file in enumerate(father_list):
        sub_list = os.listdir(input_dir + '\\' + file)
        image_data_li = []
        for band_pic in sub_list:
            width, height, proj, geotrans, band, image_data = run_merge.read_image(input_dir + '\\' + file + '\\' + band_pic)
            image_data_li.append(image_data)
        image_data_finally = np.array((image_data_li[2],image_data_li[1],image_data_li[0]),dtype=image_data_li[0].dtype)
        run_merge.write_image(output_dir + '\\' + file + '.TIF', proj, geotrans, image_data_finally)
        print(f'complete merge: {index}')
