# --*-- encoding: utf-8 --*--
import cv2
import os
import math
import shutil
import numpy as np
from scipy import ndimage
from math import *

'''
[description]
show the picture, just for debug
'''


def showImage(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)


'''
[description]
图像二值化
'''


def binarize(pic):
    ret, binary = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    return binary


'''
[description]
    pic_list: 图片的list
    pic_name: 图片名的list
    dir: 写文件的路径
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
去除图像左右白边
'''


def removePadding(pic):
    pic_median = cv2.medianBlur(pic, 5)
    cols = np.size(pic, 1)
    lines = np.size(pic, 0)
    # 去除左侧白边
    for i in range(lines):
        for j in range(cols):
            if pic_median[i][j] == 0:
                break
            pic[i][j] = 0

    # 去除右侧白边
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
对图像进行形态学操作
'''


def morphology(img, mode, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if mode == 'dilate':
        img = cv2.dilate(img, kernel)
    elif mode == 'erode':
        img = cv2.erode(img, kernel)
    elif mode == 'open':
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


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
    给定一个图像，返回最小矩形的四个点
    返回值：np.ndarray,size=(4,2)       中心坐标 (x, y)     旋转角度: [-90,0)
    '''

    def getRectangle(self, pic):
        _, contours, hierarchy = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print("contours:类型：", type(contours))
        # print("第0 个contours:", len(contours[0]))
        # print("contours amount:", len(contours))

        # 若只有一个边框，直接拿来用；否则找到边长最大的那个边框作为车票的边框
        if len(contours) > 1:
            max_distance = -9999
            target_index = -1
            for i in range(len(contours)):
                c = contours[i].squeeze()
                rect = cv2.minAreaRect(np.array(c))  # 最小拟合矩形
                box = np.int0(cv2.boxPoints(rect))  # 通过box得出矩形框
                point1 = np.array([box[0][0], box[0][1]])
                point2 = np.array([box[1][0], box[1][1]])
                distance = np.linalg.norm(point2 - point1)
                if distance > max_distance:
                    max_distance = distance
                    target_index = i
            contours = contours[target_index].squeeze()
        else:
            contours = contours[0].squeeze()
        rect = cv2.minAreaRect(np.array(contours))  # 最小拟合矩形
        box = np.int0(cv2.boxPoints(rect))  # 通过box得出矩形框

        return box, rect[0], rect[2]

    def drawLine(self, draw_img, box, color, linewwidth=2):
        cv2.line(draw_img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, linewwidth)
        cv2.line(draw_img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color, linewwidth)
        cv2.line(draw_img, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color, linewwidth)
        cv2.line(draw_img, (box[3][0], box[3][1]), (box[0][0], box[0][1]), color, linewwidth)

    '''
    [description]
    将图像进行二值化，去噪，然后获取最小拟合矩形的四个点 [ [x1,y1], ... [x4,y4] ]
    '''

    def rectangleFitting(self, img):
        raw_img = img
        img = binarize(img)  # 二值化
        img = cv2.medianBlur(img, 9)  # 中值滤波，去噪
        img = morphology(img, mode='close', kernel_size=21)  # 扩张操作
        # img = self.morphology(img, mode='dilate', kernel_size=11)           # 第二次扩张操作，使图像中心完全平滑
        img = morphology(img, mode='open', kernel_size=101)  # 开操作，去掉车票边上的小凸起

        edge = cv2.Canny(img, 50, 150)  # 边缘检测

        box, center_pos, angle = self.getRectangle(img)  # 获取最小拟合矩阵的四个角点

        # 选一个图像，画矩形框
        draw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255)
        self.drawLine(draw_img, box, color, 2)

        return draw_img, box.tolist(), center_pos, angle

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

    '''
    [description]
    旋转矩形裁剪
    '''

    def rotate(self, img, pt1, pt2, pt3, pt4):
        withRect = sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
        heightRect = sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        angle = math.acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度

        if pt4[1] > pt1[1]:  # 顺时针
            pass
        else:  # 逆时针
            angle = - angle

        height = img.shape[0]  # 原始图像高度
        width = img.shape[1]  # 原始图像宽度
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(0, 0, 0))

        # 旋转后图像的四点坐标
        [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
        [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
        [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

        # 处理反转的情况
        if pt2[1] > pt4[1]:
            pt2[1], pt4[1] = pt4[1], pt2[1]
        if pt1[0] > pt3[0]:
            pt1[0], pt3[0] = pt3[0], pt1[0]

        imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
        # cv2.imwrite('tmp.png', imgOut)
        return imgOut  # rotated image


'''
[description]
校准图像，使其成为水平正图像
'''


class Calibration(object):
    """docstring for Calibration"""

    def __init__(self, img):
        super(Calibration, self).__init__()
        self.data = self.calibrate(img)

    def calibrate(self, img):
        img = self.reshape(img)
        # img = self.flip(img)
        # row是图像行数，col是图像列数
        row = len(img)
        col = np.size(img[0])
        upper_left = img[0:int(row * 0.4), 0:int(col * 0.25)]
        lower_right = img[int(row * 0.6):row, int(col * 0.75):col]
        mean1 = np.mean(upper_left)
        mean2 = np.mean(lower_right)
        if(mean1 < mean2):
            return cv2.flip(img, -1)
        return img

    '''
    [description]
    将图像旋转成横长竖短
    '''

    def reshape(self, img):
        w, h = img.shape
        if w > h:
            return cv2.flip(img.transpose(1, 0), 1)
        else:
            return img

    '''
    [description]
    将图像旋转到正面
    '''

    def flip(self, img):
        w, h = img.shape
        left = img[:, :h // 2]
        right = img[:, h // 2:]
        if np.sum(left) > np.sum(right):
            return img
        return cv2.flip(img, -1)
