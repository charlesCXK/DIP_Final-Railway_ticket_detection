# --*-- encoding: utf-8 --*--
import cv2
import numpy as np

from dataset import DataSet
from functions import *

def main():
	train_data = DataSet(root='../train_data', mode='train', ratio=0.8).data
	# list of original pictures
	pic_ori = [cv2.imread(data[0], cv2.IMREAD_GRAYSCALE) for data in train_data]
	pic_name = [data[1] for data in train_data]


	#remove the white padding, in order to get the border of rectangle
	pic_ori = removePadding(pic_ori)

	# outline of the ticket, painted on a white background with black line
	pic_rect = getRectangle(pic_ori)
	getLines(pic_rect)

	# list of binarized pictures
	pic_bin = Binarization(pic_ori=pic_ori).pic_bin
	# writeImg(pic_bin, pic_name, '../binary_picture')

	print()


if __name__ == '__main__':
	main()