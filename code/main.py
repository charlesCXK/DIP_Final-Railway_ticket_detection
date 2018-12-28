# --*-- encoding: utf-8 --*--
import cv2
import numpy as np
from dataset import DataSet
from functions import *

''' 
[description]
对单张图像进行的一系列操作
'''
def imagePipeline(img):
    # 去除白色边框
    # img = removePadding(img)  # 处理图像周围的白块
    detectrectangle = DetectRectangle()  # 获取矩形框

    # 获取最小拟合矩形的四个点, box: [ [x1,y1], ... [x4,y4] ], 中心坐标 center_pos， 旋转角度 angle
    # https://blog.csdn.net/qq_24237837/article/details/77850496
    img_rectangle, box, center_pos, angle = detectrectangle.rectangleFitting(img)

    # 旋转并裁剪后的图像
    rotated_img = detectrectangle.rotate(img, box[0], box[1], box[2], box[3])

    # 图像配准，调整方向
    calibrated_img = Calibration(rotated_img).data

    img_num21, box21 = Num21(calibrated_img).data
    rotated_img21 = picSlim(reshape(detectrectangle.rotate(calibrated_img, box21[0], box21[1], box21[2], box21[3]), flip=2))
    return img_rectangle, calibrated_img, img_num21, rotated_img21


def main():
    images = DataSet(root='../train_data', mode='all', ratio=0.8).data

    # 原始图像的list
    pic = [cv2.imread(data[0], cv2.IMREAD_GRAYSCALE) for data in images][:]  # 测试新功能时只操作 10 张图像
    pic_name = [data[1] for data in images]

    img_rectangle, calibrated_img, img_num21, rotated_img21 = [], [], [], []
    for i in range(len(pic)):
        func_res = imagePipeline(pic[i])
        img_rectangle.append(func_res[0])
        calibrated_img.append(func_res[1])
        img_num21.append(func_res[2])
        rotated_img21.append(func_res[3])

    writeImg(img_rectangle, pic_name, '../ticket_rectangle')
    writeImg(calibrated_img, pic_name, '../ticket_calibrated')
    writeImg(img_num21, pic_name, '../ticket_num21')
    writeImg(rotated_img21, pic_name, '../ticket_num21_rotated')


if __name__ == '__main__':
    main()
