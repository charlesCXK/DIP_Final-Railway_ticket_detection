# --*-- encoding: utf-8 --*--
import cv2
import numpy as np

from dataset import DataSet
from functions import Binarization

def main():
	train_data = DataSet(root='../train_data', mode='train', ratio=0.8).data

if __name__ == '__main__':
	main()