# --*-- encoding: utf-8 --*--
# This file supplies some useful functions
 
import sys
import math
import numpy as np

# 文件夹路径要按照调用 functions.py 的路径来
ANNOTATION = '../annotation.txt'

'''
[description]
获得车票标注，返回值为字典 {pic_name: annoation}
'''
def get_annotation(filename):
	annotation = {}
	with open(filename, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	for line in lines:
		seg = line.strip().split()
		if len(seg) <= 0:
			continue
		annotation[seg[0]] = seg[1]
	return annotation

'''
[description]
获取训练集/测试集列表
'''
def get_train_test(ratio):
	with open('../annotation.txt', 'r', encoding='utf-8') as f:
		lines = f.readlines()
	image_list = [lines[i].strip().split()[0] for i in range(len(lines)) if lines[i].strip()]
	labels = get_annotation(ANNOTATION)

	train_num = math.ceil(len(image_list) * ratio)

	train_set = list(np.random.choice(image_list, train_num, replace=False))
	train_label = [labels[item] for item in train_set]
	test_set = [item for item in image_list if item not in train_set]
	test_label = [labels[item] for item in test_set]


	return train_set, train_label, test_set, test_label
	


