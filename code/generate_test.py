import os
import shutil
import cv2
from dataset import DataSet
images = DataSet(root='../train_data', mode='test', ratio=0.8).data
print(images[0])

testdir = './test_data'
if (os.path.exists(testdir) == 0):
    os.mkdir(testdir)
else:
    shutil.rmtree(testdir)
    os.mkdir(testdir)

for img_path, img_name, img_label in images:
	img = cv2.imread(img_path)
	cv2.imwrite(os.path.join(testdir, img_name), img)

'''
[description]
生成图像路径
'''
with open('annotation.txt', 'w', encoding='utf-8') as f:
    for i in range(len(images)):
        f.write(images[i][0].split('/')[-1] + '\n')

'''
[description]
生成标准标签
'''
with open('groundtruth.txt', 'w', encoding='utf-8') as f:
    for i in range(len(images)):
        f.write(images[i][2] + '\n')