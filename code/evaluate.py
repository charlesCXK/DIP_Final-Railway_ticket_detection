# --*-- encoding: utf-8 --*--
# Classifier.py
# 进行数字与字母的识别
import cv2
import os
import shutil
import math
import numpy as np

from sklearn import svm
from mainprocess import imagePipeline

alphabet = [chr(i) for i in range(65,91)]        # 字母表


class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, root, ratio, classnum, C=1, predict=False):
        super(Classifier, self).__init__()
        self.root = root
        self.ratio = ratio
        self.classnum = classnum
        self.traind, self.trainl, self.testd, self.testl = self.make_data()

        self.clf = svm.SVC(kernel='linear', C=C)
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

# 获取测试集图像的路径
with open('annotation.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
image_path = [line.strip() for line in lines if len(line.strip())>0]

saveDir = 'segments'
testRoot = '../train_data'
f = open('prediction.txt', 'w', encoding='utf-8')       # 预测结果

if (os.path.exists(saveDir) == 0):
    os.mkdir(saveDir)
else:
    shutil.rmtree(saveDir)
    os.mkdir(saveDir)

# 处理每张图像
for i in range(len(image_path)):
    img = cv2.imread(os.path.join(testRoot, image_path[i]), cv2.IMREAD_GRAYSCALE)     # 读取图像
    func_res = imagePipeline(img)
    seg_list, img_num_all = func_res[2], func_res[3]

    cv2.imwrite(os.path.join(saveDir, image_path[i]), img_num_all)

    pred_str21 = ''
    pred_str7 = ''

    for j in range(21):

        data = seg_list[j].flatten()        # 数据展开
        data = (data>0).astype(np.int8)
        data = data.reshape(1, -1)          # 增加一个维度
        if j == 14:
            pred = letter_classify.predict(data)
        else:
            pred = number_classify.predict(data)
        pred_number = pred[0]
        if j != 14:
            pred_str21 = pred_str21 + str(pred_number)
        else:
            pred_str21 = pred_str21 + alphabet[pred_number]

    pred_str7 = pred_str21[-7:]
    f.write(image_path[i] + ' ' + pred_str21 + ' ' + pred_str7 + '\n')
    
f.close()
