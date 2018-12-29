# --*-- encoding: utf-8 --*--
# Classifier.py
# 进行数字与字母的识别
import cv2
import os
import math
import numpy as np

from sklearn import svm


class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, root, ratio, classnum, C=1, predict=False):
        super(Classifier, self).__init__()
        self.root = root
        self.ratio = ratio
        self.classnum = classnum
        self.traind, self.trainl, self.testd, self.testl = self.make_data()

        self.clf = svm.SVC(kernel='linear', C=0.4)
        self.clf = self.train(self.traind, self.trainl)       

        if predict: 
            self.predict(self.testd, self.testl)

    '''
    [description]
    将标签进行独热编码
    '''
    def OneHotEnc(self, label, classnum):
        enc = np.zeros(classnum)
        enc[label] = 1
        return enc

    '''    
    [description]
    制造数据集
    '''
    def make_data(self):
        file = 'labels.txt'

        # 读入 labels 文件
        with open(os.path.join(self.root, file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.strip().split() for line in lines if len(line.strip())>0]

        # 训练集数量
        train_num = math.ceil(len(lines) * self.ratio)

        # 训练集和测试集的编号
        train_set = list(np.random.choice(range(len(lines)), train_num, replace=False))
        test_set = [item for item in range(len(lines)) if item not in train_set]
        
        '''
        train_data: 训练集向量
        train_label: 训练集标签 (int)
        '''
        train_data = [cv2.imread(os.path.join(self.root, lines[item][0]), cv2.IMREAD_GRAYSCALE).flatten() for item in train_set]
        train_label = [int(lines[item][1]) for item in train_set]
        test_data = [cv2.imread(os.path.join(self.root, lines[item][0]), cv2.IMREAD_GRAYSCALE).flatten() for item in test_set]
        test_label = [int(lines[item][1]) for item in test_set]
        # train_data = train_data*4
        # train_label = train_label*4
        return (np.array(train_data)>0).astype(np.int8), np.array(train_label), (np.array(test_data)>0).astype(np.int8), np.array(test_label)

    def train(self, x, y):
        self.clf.fit(x, y)
        return self.clf

    def predict(self, x, y):
        pred = self.clf.predict(x)
        gt = y
        res = np.array(pred) == y
        print('accuracy is {}'.format(np.sum(res)/res.shape[0]))

# 定义分类器
number_classify = Classifier('../number_data', 1, 10, C=0.4, predict=False).clf
letter_classify = Classifier('../letter_data', 1, 26, C=1, predict=False).clf  
