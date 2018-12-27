import cv2
import numpy as np
from dataset import DataSet
from functions import *

if __name__ == '__main__':
    images = DataSet(root='../ticket_calibrated', mode='all', ratio=1).data
    pic = [cv2.imread(data[0], cv2.IMREAD_GRAYSCALE) for data in images][:]  # 只操作 10 张图像
    pic_name = [data[1] for data in images]
    # binarized_img = [binarize(p, 50) for p in pic]
    # writeImg(binarized_img, pic_name, '../ticket_binarized')
    # img_num21 = []
    # for i in range(len(pic)):
    #     # print(pic_name[i])
    #     img_num21.append(Num21(pic[i], pic_name[i]).data)

    img_num21 = [Num21(p).data for p in pic]
    writeImg(img_num21[0], pic_name, '../ticket_num21')