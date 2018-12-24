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
	img = removePadding(img)		# 处理图像周围的白块
	img, box = DetectRectangle().rectangleFitting(img)		# 获取最小拟合矩形的四个点, box: [ [x1,y1], ... [x4,y4] ]
	return img

def main():
	images = DataSet(root='../train_data', mode='all', ratio=0.8).data

	# list of original pictures
	pic = [cv2.imread(data[0], cv2.IMREAD_GRAYSCALE) for data in images]
	pic_name = [data[1] for data in images]

	pic = [imagePipeline(p) for p in pic[:10]]		# 只操作 10 张图像
	writeImg(pic, pic_name, '../ticket_rectangle')




if __name__ == '__main__':
	main()