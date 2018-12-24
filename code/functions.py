# --*-- encoding: utf-8 --*--
import cv2
import os
import math
import shutil
import numpy as np
from scipy import ndimage


'''
[description]
show the picture, just for debug
'''
def showImage(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

'''
[description]
binarize the pictures
'''
def binarize(pic):
    ret, binary = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    return binary

'''
[description]
    pic_list: list of image array
    pic_name: list of image name
    dir: name of directory
'''
def writeImg(pic_list, pic_name, dir):
    if (os.path.exists(dir) == 0):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)
    for i in range(len(pic_list)):
        cv2.imwrite(os.path.join(dir, pic_name[i]), pic_list[i])

'''
[description]
remove the zero padding on the left and right
'''
def removePadding(pic):
    pic_median = cv2.medianBlur(pic, 5)
    cols = len(pic[0])
    lines = int(np.size(pic) / cols)
    # mark the column of zero padding on the left
    for i in range(lines):
        for j in range(cols):
            if pic_median[i][j] == 0:
                break
            pic[i][j] = 0

    # mark the column of zero padding on the right
    for i in range(lines):
        for j in range(cols):
            if pic_median[i][cols - j - 1] == 0:
                pad = cols - j
                break
    for i in range(lines):
        for j in range(pad, cols):
            pic[i][j] = 0
    return pic

'''
[description]
检测车票票面，返回矩形框的四个点
'''
class DetectRectangle(object):
    """docstring for DetectRectangle"""
    def __init__(self):
        super(DetectRectangle, self).__init__()

    '''
    [description]
    对图像进行形态学操作
    '''
    def morphology(self, img, mode, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if mode == 'dilate':
            img = cv2.dilate(img, kernel)
        elif mode == 'erode':
            img =cv2.erode(img,kernel)
        elif mode == 'open':
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img

    '''
    [description]
    给定一个图像，返回最小矩形的四个点
    返回值：np.ndarray, size=(4,2)
    '''
    def getRectangle(self, pic):
        _, contours, hierarchy = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print("contours:类型：", type(contours))
        # print("第0 个contours:", len(contours[0]))
        # print("contours 数量：", len(contours))

        contours = contours[0].squeeze()
        rect = cv2.minAreaRect(np.array(contours))      # 最小拟合矩形
        box = np.int0(cv2.boxPoints(rect))  # 通过box得出矩形框

        return box

    '''
    [description]
    将图像进行二值化，去噪，然后获取最小拟合矩形的四个点 [ [x1,y1], ... [x4,y4] ]
    '''
    def rectangleFitting(self, img):
        raw_img = img
        img = binarize(img)     # 二值化
        img = cv2.medianBlur(img, 9)        # 中值滤波，去噪
        img = self.morphology(img, mode='dilate', kernel_size=21)           # 扩张操作
        img = self.morphology(img, mode='dilate', kernel_size=21)           # 第二次扩张操作，使图像中心完全平滑
        img = self.morphology(img, mode='open', kernel_size=101)             # 开操作，去掉车票边上的小凸起

        edge = cv2.Canny(img, 50, 150)      # 边缘检测

        box = self.getRectangle(img)         # 获取最小拟合矩阵的四个角点

        cv2.line(edge, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (255, 0, 0), 5)
        cv2.line(edge, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (255, 0, 0), 5)
        cv2.line(edge, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (255, 0, 0), 5)
        cv2.line(edge, (box[3][0], box[3][1]), (box[0][0], box[0][1]), (255, 0, 0), 5)

        return edge, box.tolist()

        # lines = cv2.HoughLinesP(edge, 0.8, np.pi / 180, 90,
        #                         minLineLength=50, maxLineGap=300)
        #
        # # 统计斜率
        # slope = []
        # for i in range(lines.shape[0]):
        #     for x1, y1, x2, y2 in lines[i]:
        #         if x1 == x2:
        #             this_slope = 100
        #         else:
        #             this_slope = (y2-y1)/(x2-x1)
        #         slope.append(this_slope)
        # slope1 = [s for s in slope if abs(s)>10]
        # slope2 = [s for s in slope if abs(s) < 1]
        #
        # mean_slope1 = sum(slope1)/len(slope1)
        # mean_slope2 = sum(slope2) / len(slope2)
        #
        # angle = math.atan(mean_slope2) * 180 / np.pi
        # rotated = ndimage.rotate(raw_img, 90+angle)
        # return rotated

