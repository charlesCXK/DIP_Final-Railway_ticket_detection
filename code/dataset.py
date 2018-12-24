# --*-- encoding: utf-8 --*--
import os
import math
import numpy as np

ANNOTATION = '../annotation.txt'

class DataSet(object):
	"""get rada"""
	def __init__(self, root, mode, ratio):
		super(DataSet, self).__init__()
		self.root = root
		self.mode = mode
		self.data = self.make_dataset(root, mode, ratio)

	'''
	[description]
	获得车票标注，返回值为字典 {pic_name: annoation}
	'''
	def get_annotation(self, filename):
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
	def get_train_test(self, ratio):
		with open(ANNOTATION, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		image_list = [lines[i].strip().split()[0] for i in range(len(lines)) if lines[i].strip()]
		labels = self.get_annotation(ANNOTATION)

		train_num = math.ceil(len(image_list) * ratio)

		train_set = list(np.random.choice(image_list, train_num, replace=False))
		train_label = [labels[item] for item in train_set]
		test_set = [item for item in image_list if item not in train_set]
		test_label = [labels[item] for item in test_set]

		return train_set, train_label, test_set, test_label

	'''
	[description]
	根据 训练/测试集，返回数据的可直接读取的路径、图片名、标签
	'''
	def make_dataset(self, root, mode, ratio):
		assert mode in ['train', 'test', 'all']
		train_data, train_label, test_data, test_label = self.get_train_test(ratio)
		data_list = []

		if mode == 'train':
			data_path = train_data
			label = train_label
		elif mode == 'test':
			data_path = test_data
			label = test_label
		else:
			data_path = train_data + test_data
			label = train_label + test_label


		for i in range(len(data_path)):
			data_list.append([os.path.join(root, data_path[i]), data_path[i], label[i]])
		return data_list