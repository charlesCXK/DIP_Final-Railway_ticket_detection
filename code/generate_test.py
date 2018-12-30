from dataset import DataSet
images = DataSet(root='../train_data', mode='test', ratio=0.8).data
print(images[0])

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