# --*-- encoding: utf-8 --*--
import cv2
import numpy as np
from dataset import DataSet
from functions import *

alphabet = [chr(i) for i in range(65,91)]        # 字母表

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

    # 获取 21 位码
    img_num21, box21 = Num21(calibrated_img).data
    rotated_img21 = picSlim(reshape(detectrectangle.rotate(calibrated_img, box21[0], box21[1], box21[2], box21[3]), flip=2), 2)

    seg_list = SegElement21(rotated_img21).rt

    # 获取 7 位码
    img_num7, box7 = Num7(calibrated_img).data
    rotated_img7 = picSlim(reshape(detectrectangle.rotate(calibrated_img, box21[0], box21[1], box21[2], box21[3]), flip=2), 2)

    img_num_all = NumAll(calibrated_img, box21, box7).data

    return img_rectangle, calibrated_img, seg_list, img_num_all


def main():
    images = DataSet(root='../train_data', mode='all', ratio=0.8).data

    # 原始图像的list
    pic = [cv2.imread(data[0], cv2.IMREAD_GRAYSCALE) for data in images][:10] # 测试新功能时只操作 10 张图像
    pic_name = [data[1] for data in images]
    pic_label21 = [data[2] for data in images]

    img_rectangle, calibrated_img, img_num21, rotated_img21, number_list, letter_list, img_num7, rotated_img7, img_num_all = [], [], [], [], [], [],[], [], []
    number_label, letter_label = [], []
    number_name, letter_name = [], []
    for i in range(len(pic)):
        func_res = imagePipeline(pic[i])
        img_rectangle.append(func_res[0])
        calibrated_img.append(func_res[1])
        img_num_all.append(func_res[3])

        label21 = pic_label21[i]        # 该图片的21位码标签
        seg_list = func_res[2]
        for j in range(21):
            if j != 14:
                number_list.append(seg_list[j])
                number_label.append(int(label21[j]))
            else:
                letter_list.append(seg_list[j])
                letter_label.append(alphabet.index(label21[j]))

    # 为每个元素定义一个名字
    for i in range(len(number_list)):
        number_name.append('{}.png'.format(i))
    for i in range(len(letter_list)):
        letter_name.append('{}.png'.format(i))


    writeImg(img_rectangle, pic_name, '../ticket_rectangle')
    writeImg(calibrated_img, pic_name, '../ticket_calibrated')
    # 这两个图片list是中间过程的结果，最终不需要写到硬盘
    # writeImg(img_num21, pic_name, '../ticket_num21')
    # writeImg(img_num7, pic_name, '../ticket_num7')
    writeImg(img_num_all, pic_name, '../ticket_num_all')
    writeImg(number_list, number_name, '../number_data')
    writeImg(letter_list, letter_name, '../letter_data')

    # 写单张图的标签
    with open('../number_data/labels.txt', 'w', encoding='utf-8') as f:
        for i in range(len(number_list)):
            f.write('{}.png\t{}\n'.format(i, number_label[i]))
    with open('../letter_data/labels.txt', 'w', encoding='utf-8') as f:
        for i in range(len(letter_list)):
            f.write('{}.png\t{}\n'.format(i, letter_label[i]))


if __name__ == '__main__':
    main()
