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
def showImage(img, img_name='Image'):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
[description]
图像二值化
'''
def binarize(pic, threshold=127):
    ret, binary = cv2.threshold(pic, threshold, 255, cv2.THRESH_BINARY)
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
将图像旋转成横长竖短
'''
def reshape(img, flip=1):
    w, h = img.shape
    if w > h:
        if flip==2:     # 逆时针旋转 90°
            return cv2.flip(cv2.flip(img, -1).transpose(1, 0), 1)
        else:
            return cv2.flip(img.transpose(1, 0), flip)
    else:
        return img

'''
[description]
删除图片左右白边
'''
def picSlim(img, left_border = 0, right_border = 0):
    img_bin = binarize(img, 80)
    sum_col = np.sum(img_bin, 0)
    sum_row = np.sum(img_bin, 1)
    row = len(img)
    col = np.size(img[0])
    lim_col = row * 255
    lim_row = col * 255
    left_padding = 0
    right_padding = 0
    upper_padding = 0
    lower_padding = 0
    for i in range(col):
        # 该列中有黑字
        if(sum_col[i] < lim_col):
            break
        else:
            left_padding += 1
    for i in range(col):
        # 该列中有黑字
        if(sum_col[col-i-1] < lim_col):
            break
        else:
            right_padding += 1
    for i in range(row):
        # 该行中有黑字
        if (sum_row[i] < lim_row):
            break
        else:
            upper_padding += 1
    for i in range(row):
        # 该列中有黑字
        if(sum_row[row-i-1] < lim_row):
            break
        else:
            lower_padding += 1
    img_ret = img[upper_padding:row-lower_padding, left_padding-left_border:col-right_padding+right_border]
    return img_ret

'''
[description]
给定划分的个数以及box，在图上画出分割线
'''
def drawSegLines(img, seg_num, box):
    # 重新排列box，使得四个顶点按照左下角，左上角，右上角，右下角的顺序排列
    box_relist = []
    box_new = []
    for i in range(len(box)):
        box_relist.append([box[i][0], box[i][1], box[i][0] + box[i][1]])
    min_num = 10000
    # 斜率一般不大，因此行列坐标和最小的是左下角，即box_new[0]
    for i in range(4):
        if box_relist[i][2] < min_num:
            min_index = i
            min_num = box_relist[i][2]
    box_new.append([box_relist[min_index][0], box_relist[min_index][1]])
    del (box_relist[min_index])

    min_num = 10000
    # box_new[1]
    for i in range(3):
        if abs(box_relist[i][0] - box_new[0][0]) < min_num:
            min_index = i
            min_num = abs(box_relist[i][0] - box_new[0][0])
    box_new.append([box_relist[min_index][0], box_relist[min_index][1]])
    del (box_relist[min_index])

    min_num = 10000
    # box_new[2]
    for i in range(2):
        if abs(box_relist[i][1] - box_new[1][1]) < min_num:
            min_index = i
            min_num = abs(box_relist[i][1] - box_new[1][1])
    box_new.append([box_relist[min_index][0], box_relist[min_index][1]])
    del (box_relist[min_index])

    # box_new[3]
    box_new.append([box_relist[0][0], box_relist[0][1]])

    gradient = (box_new[2][1] - box_new[1][1]) / (box_new[2][0] - box_new[1][0])

    color = (0, 0, 255)
    linewidth = 1
    if img.ndim == 2:
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        draw_img = img
    detectrectangle = DetectRectangle()
    detectrectangle.drawLine(draw_img, box_new, color, linewidth)

    # 21位码分割
    if seg_num == 21:
        # box_new[0][0] += -3
        # box_new[1][0] += -3

        # 划分比例，字母大约是数字的1.6倍
        col_distance = box_new[2][0] - box_new[1][0]
        x1 = box_new[0][0]
        y1 = box_new[0][1]
        x2 = box_new[1][0]
        y2 = box_new[1][1]

        for i in range(1, seg_num):
            # 字母比数字大
            if i < 15:
                x3 = x1 + int(i * col_distance / 21.45)
                x4 = x2 + int(i * col_distance / 21.45)
            if i >= 15:
                x3 = x1 + int((i + 0.45) * col_distance / 21.45)
                x4 = x2 + int((i + 0.45) * col_distance / 21.45)
            y3 = y1 + int((x3 - x1) * gradient)
            y4 = y2 + int((x4 - x2) * gradient)
            cv2.line(draw_img, (x3, y3), (x4, y4), color, linewidth)
    # 7位码分割
    else:
        # if(box_new[0][0] <= 3):
        #     box_new[0][0] = 3
        # if(box_new[1][0] <= 3):
        #     box_new[1][0] = 3
        # box_new[0][0] += -3
        # box_new[1][0] += -3

        # 划分比例，字母大约是数字的1.6倍
        col_distance = box_new[2][0] - box_new[1][0]
        x1 = box_new[0][0]
        y1 = box_new[0][1]
        x2 = box_new[1][0]
        y2 = box_new[1][1]

        for i in range(1, seg_num):
            # 字母比数字大
            x3 = x1 + int((i + 0.45) * col_distance / 7.45)
            x4 = x2 + int((i + 0.45) * col_distance / 7.45)
            y3 = y1 + int((x3 - x1) * gradient)
            y4 = y2 + int((x4 - x2) * gradient)
            cv2.line(draw_img, (x3, y3), (x4, y4), color, linewidth)
    return draw_img

'''
[description]
找到二维码的位置，返回二维码矩形框四个顶点的box数组
'''
def findBitcode(img):
    # 先二值化去除原图中阴影部分
    binarized_img = binarize(img, 20)
    row = len(img)
    col = np.size(img[0])

    # 截取右下角二维码部分，取反并做闭操作
    lower_right0 = binarized_img[int(row * 0.55):row, int(col * 0.72):col]
    lower_right0 = cv2.bitwise_not(lower_right0)
    col_bitcode_ori = np.size(lower_right0[0])
    col_bitcode_ori = int(col_bitcode_ori / 2)
    # 由于有二维码缺损的异常数据，只取左边一半
    lower_right1 = lower_right0[:, 0:col_bitcode_ori]
    lower_right1 = morphology(lower_right1, 'close', 21)

    row_bitcode = len(lower_right1)
    col_bitcode = np.size(lower_right1[0])
    # 去掉右下角圆角和矩形外边缘空隙
    # img_zero = np.zeros((40, 40))
    # img_zero = img_zero.astype(np.uint8)
    # lower_right1[row_bitcode-40:row_bitcode, col_bitcode-40:col_bitcode] = img_zero
    #
    # # 删除右侧白边
    # row_sub = len(lower_right1)
    # col_sub = np.size(lower_right1[0])
    # sum_col = np.sum(lower_right1, 0)
    # del_col = 0
    # for i in range(col_sub):
    #     # 多余白字部分
    #     if sum_col[col_sub-i-1] > 255:
    #         del_col += 1
    #     else:
    #         break
    # lower_right2 = np.zeros((row_sub, col_sub - del_col))
    # lower_right2 = lower_right1[:, 0:col_sub-del_col]

    # 删除底部白边
    row_sub = len(lower_right1)
    col_sub = np.size(lower_right1[0])
    del_row = 0
    sum_row = np.sum(lower_right1, 1)
    for i in range(row_sub):
        if sum_row[row_sub - i - 1] > 255:
            del_row += 1
        else:
            break
    lower_right2 = lower_right1
    lower_right2[row_sub - del_row:row_sub, :] = 0

    # 在黑色背景中对应位置放入左半边二维码
    img_res = np.zeros((row, col))
    img_res = img_res.astype(np.uint8)
    img_res[row - row_bitcode:row, col - col_bitcode - col_bitcode_ori:col - col_bitcode_ori] = lower_right2
    # showImage(img_res)

    # 在上图中圈出二维码所在矩形框
    detectrectangle = DetectRectangle()
    box, center_pos, angle = detectrectangle.getRectangle(img_res)
    # print(box)
    return box


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
        
        try:
            # 若只有一个边框，直接拿来用；否则找到边长最大的那个边框作为车票的边框
            if len(contours) > 1:
                max_distance = -9999
                target_index = -1
                for i in range(len(contours)):
                    c = contours[i].squeeze()
                    if(len(c) < 3):
                        continue
                    rect = cv2.minAreaRect(np.array(c))  # 最小拟合矩形
                    box = np.int0(cv2.boxPoints(rect))  # 通过box得出矩形框
                    point1 = np.array([box[0][0], box[0][1]])
                    point2 = np.array([box[1][0], box[1][1]])
                    point3 = np.array([box[2][0], box[2][1]])
                    distance = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point2)
                    if distance > max_distance:
                        max_distance = distance
                        target_index = i
                contours = contours[target_index].squeeze()
            else:
                contours = contours[0].squeeze()
        except:
            showImage(pic)
            exit()
        rect = cv2.minAreaRect(np.array(contours))  # 最小拟合矩形
        box = np.int0(cv2.boxPoints(rect))  # 通过box得出矩形框

        return box, rect[0], rect[2]

    '''
    [description]
    给定图片和四个顶点，在图中画出矩形
    '''
    def drawLine(self, draw_img, box, color=(0, 0, 255), linewidth=2):
        cv2.line(draw_img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, linewidth)
        cv2.line(draw_img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color, linewidth)
        cv2.line(draw_img, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color, linewidth)
        cv2.line(draw_img, (box[3][0], box[3][1]), (box[0][0], box[0][1]), color, linewidth)

    '''
    [description]
    将图像进行二值化，去噪，然后获取最小拟合矩形的四个点 [ [x1,y1], ... [x4,y4] ]
    '''
    def rectangleFitting(self, img):
        raw_img = img
        img = binarize(img)  # 二值化
        img = cv2.medianBlur(img, 9)  # 中值滤波，去噪
        # img = morphology(img, mode='close', kernel_size=21)  # 扩张操作
        # # img = self.morphology(img, mode='dilate', kernel_size=11)           # 第二次扩张操作，使图像中心完全平滑
        # img = morphology(img, mode='open', kernel_size=101)  # 开操作，去掉车票边上的小凸起

        img = morphology(img, mode='close', kernel_size=21)  # 扩张操作
        img = morphology(img, mode='open', kernel_size=81)  # 开操作，去掉车票边上的小凸起
        cv2.imwrite('./1.png', img)


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

    '''
    [description]
    将车票旋转成正常角度（即二维码在右下角的方向，并且横长竖短），并用白底补全缺损的图像
    '''
    def calibrate(self, img):
        pic = reshape(img)
        pic = self.turnToCorrect(pic)
        pic = self.completion(pic)
        return pic

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

    '''
    [description]
    将横长竖短的车票旋转成正常角度（即二维码在右下角的方向）
    '''
    def turnToCorrect(self, img):
        # row是图像行数，col是图像列数
        row = len(img)
        col = np.size(img[0])
        # 左上角和右下角
        upper_left = img[0:int(row * 0.4), 0:int(col * 0.25)]
        lower_right = img[int(row * 0.6):row, int(col * 0.75):col]
        # 统计左上角和右下角灰度平均值，如果左上角均值较小，证明二维码在左上角，要旋转180°
        mean1 = np.mean(upper_left)
        mean2 = np.mean(lower_right)
        if (mean1 < mean2):
            return cv2.flip(img, -1)
        return img

    '''
    [description]
    探测图像长宽比，如有明显异常的，用白底补全缺损的部分
    '''
    def completion(self, img):
        row = len(img)
        col = np.size(img[0])
        # 正常图像长宽比约为1.61
        aspect_ratio = col / row
        # 图像有缺损
        if (aspect_ratio < 1.55):
            lower_right = img[int(row * 0.7):int(row * 0.8), col - 20:col]
            mean1 = np.mean(lower_right)
            col_new = int(row * 1.61)
            # img_new = np.zeros(shape=(row, col_new))
            # blank_array = []
            # for i in range(col_new):
            #     blank_array.append(255)
            # 二维码部分是否有缺损
            np_blank = np.ones((row, col_new)) * 255
            if (mean1 > 127):  # 右下角区域偏亮，二维码有缺损，补全到右边
                np_blank[:, col_new - col:col_new] = img
                # img_new[i] = np_blank
            else:  # 二维码无缺损，补全到左边
                np_blank[:, 0:col] = img
            np_blank = np_blank.astype(np.uint8)
            return np_blank
        return img


'''
[description]
获取21位码
'''
class Num21(object):
    """docstring for Num21"""
    
    # pic_name是调试时用的参数，方便找到分割出错的图片
    def __init__(self, img, pic_name='0'):
        super(Num21, self).__init__()
        self.data = self.num21(img, pic_name)

    def num21(self, img, pic_name):
        # 先二值化去除原图中阴影部分
        binarized_img = binarize(img, 20)
        row = len(img)
        col = np.size(img[0])
        
        box = findBitcode(img)

        '''
        二维码高4.5
        21位码右边距离二维码左边9.3
        21位码顶边距离二维码底边0.3
        21位码长10.4，高1.0
        二维码的高度固定，以它为标准
        '''
        row_bitcode_upper = min(b[1] for b in box)
        row_bitcode_lower = max(b[1] for b in box)
        col_bitcode_left = min(b[0] for b in box)
        col_bitcode_right = max(b[0] for b in box)
        row_bitcode_distance = row_bitcode_lower - row_bitcode_upper
        row_num21_upper = row_bitcode_lower + int(0.3/4.5*row_bitcode_distance)
        row_num21_lower = row_bitcode_lower + int(1.6/4.5*row_bitcode_distance)
        col_num21_right = col_bitcode_left - int(9.3/4.5*row_bitcode_distance)
        col_num21_left = col_bitcode_left - int(20.7/4.5*row_bitcode_distance)

        # # 在原图中大致画出21位码矩形框
        # box_old = np.zeros((4, 2))
        # box_old[0] = [col_num21_left, row_num21_lower]
        # box_old[1] = [col_num21_left, row_num21_upper]
        # box_old[2] = [col_num21_right, row_num21_upper]
        # box_old[3] = [col_num21_right, row_num21_lower]
        # box_old = box_old.astype(np.int64)
        # draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # color = (0, 0, 255)
        # detectrectangle.drawLine(draw_img, box_old, color, 2)

        # 取出二值化图中上一步圈出的矩形框区域
        row_sub = row_num21_lower - row_num21_upper
        col_sub = col_num21_right - col_num21_left
        num21_rect = np.zeros((row_sub, col_sub))
        num21_rect = binarized_img[row_num21_upper:row_num21_lower, col_num21_left:col_num21_right]
        num21_rect = cv2.bitwise_not(num21_rect)
        # tmp_num21_rect = num21_rect.copy()

        try:
            num21_rect = morphology(num21_rect, 'close', 21)
            num21_rect = morphology(num21_rect, 'dilate', 7)
        except:
            print('cnm')
            cv2.imwrite('./tmp.png', binarized_img)
            print(row_num21_upper, row_num21_lower, col_num21_left, col_num21_right)
            
        img_res = np.zeros((row, col))
        img_res = img_res.astype(np.uint8)
        img_res[row_num21_upper:row_num21_lower, col_num21_left:col_num21_right] = num21_rect

        # 在原图中画出21位码矩形框
        detectrectangle = DetectRectangle()
        box_new, center_pos, angle = detectrectangle.getRectangle(img_res)
        # draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.line(draw_img, (box_new[0][0], box_new[0][1]), (box_new[1][0], box_new[1][1]), (0, 0, 255), 2)
        # color = (0, 0, 255)
        # detectrectangle.drawLine(draw_img, box_new, color, 1)
        draw_img = drawSegLines(img, 21, box_new)
        return [draw_img, box_new]



'''
[description]
给定小图，分割出其中的数字与字母
'''
class SegElement21(object):
    """docstring for SegElement21"""
    def __init__(self, img):
        super(SegElement21, self).__init__()
        self.img = img
        self.rt = self.segImage(img)

    '''
    [description]
    返回分割后小图的列表
    '''
    def segImage(self, img):        
        img = cv2.resize(img,(324, 30),interpolation=cv2.INTER_LINEAR)
        h, w = img.shape
        unit_scale = round(w / 21.6) # 平均一个单位元素的长度
        seglist = self.generateSegDelta(unit_scale)     # 获取分割下标

        data_list = []

        # 数字的阈值
        number_img = binarize(img, 60)
        # 字母的阈值
        letter_img = binarize(img, 80)

        for i in range(len(seglist)):
            if i != 14:
                data_list.append(number_img[:, seglist[i][0]: seglist[i][1]])
            else:
                data_list.append(letter_img[:, seglist[i][0]: seglist[i][1]])
        return data_list

    '''
    [description]
    生成每个元素的起始下标与结束下标
    '''
    def generateSegDelta(self, unit_scale):
        seglist = []
        for i in range(14):
            seglist.append([round(i*unit_scale), round(i*unit_scale+unit_scale)])
        seglist.append([round(14*unit_scale), round(14*unit_scale)+round(unit_scale*1.6)])
        for i in range(6):
            seglist.append([round(14*unit_scale)+round(unit_scale*1.6)+round(i*unit_scale), round(14*unit_scale)+round(unit_scale*1.6)+round(i*unit_scale)+unit_scale])
        return seglist
    

'''
[description]
获取7位码
'''
class Num7(object):
    """docstring for Num7"""
    def __init__(self, img, pic_name='0'):
        super(Num7, self).__init__()
        self.data = self.num7(img, pic_name)

    def num7(self, img, pic_name):

        # 先二值化去除原图中噪声，七位码的灰度值在60~120之间
        binarized_img = (img > 60).astype(np.uint8) & (img < 120).astype(np.uint8)
        binarized_img = binarized_img.astype(np.uint8) * 255
        row = len(img)
        col = np.size(img[0])
        # 定位二维码
        box = findBitcode(img)

        '''
        二维码高4.5
        7位码右边距离二维码左边14.8
        7位码底边距离二维码底边8.9
        7位码长5.7，高1.1
        但7位码的位置偏差较大，需要框出更大的范围
        二维码的高度固定，以它为标准
        '''
        row_bitcode_upper = min(b[1] for b in box)
        row_bitcode_lower = max(b[1] for b in box)
        col_bitcode_left = min(b[0] for b in box)
        col_bitcode_right = max(b[0] for b in box)
        row_bitcode_distance = row_bitcode_lower - row_bitcode_upper
        row_num7_upper = row_bitcode_upper - int(10.7 / 4.5 * row_bitcode_distance)
        row_num7_lower = row_bitcode_upper - int(7.3 / 4.5 * row_bitcode_distance)
        col_num7_right = col_bitcode_left - int(13 / 4.5 * row_bitcode_distance)
        col_num7_left = col_bitcode_left - int(22.2 / 4.5 * row_bitcode_distance)
        if row_num7_upper < 0:
            row_num7_upper = 0
        if col_num7_left < 0:
            col_num7_left = 0

        # 取出二值化图中上一步圈出的矩形框区域
        row_sub = row_num7_lower - row_num7_upper
        col_sub = col_num7_right - col_num7_left
        num7_rect = np.zeros((row_sub, col_sub))
        num7_rect = binarized_img[row_num7_upper:row_num7_lower, col_num7_left:col_num7_right]

        #
        # if(pic_name == '2018-5-22-20-0-13.bmp' or pic_name == '2018-5-22-19-55-2.bmp'):
        #     showImage(num7_rect, pic_name)
        #     for i in range(row_sub):
        #         print(i)
        #         print(num7_rect[i])
        #         print()

        # 票面中汉字的边缘和七位码灰度值在同一范围，利用均值滤波+阈值分割去掉汉字边缘部分
        num7_rect = cv2.blur(num7_rect, (3, 3))
        num7_rect = binarize(num7_rect, 86)
        # 现在汉字边缘仍有残留的白点，利用中值滤波去除这些盐噪声
        num7_rect = cv2.medianBlur(num7_rect, 3)
        num7_rect = cv2.medianBlur(num7_rect, 3)
        num7_rect = morphology(num7_rect, 'close', 19)
        num7_rect = morphology(num7_rect, 'dilate', 11)

        # for i in range(row_sub):
        #     print(i)
        #     print(num7_rect[i])
        #     print()
        # showImage(num7_rect)
        # num7_rect = img[row_num7_upper:row_num7_lower, col_num7_left:col_num7_right]
        # showImage(num7_rect)
        # np.set_printoptions(threshold=np.NaN)
            # print()
        # num7_rect = cv2.bitwise_not(num7_rect)
        # num7_rect = morphology(num7_rect, 'close', 15)
        # num7_rect = morphology(num7_rect, 'dilate', 7)
        # showImage(num7_rect)

        img_res = np.zeros((row, col))
        img_res = img_res.astype(np.uint8)
        img_res[row_num7_upper:row_num7_lower, col_num7_left:col_num7_right] = num7_rect

        # 在原图中画出7位码矩形框
        detectrectangle = DetectRectangle()
        box_new, center_pos, angle = detectrectangle.getRectangle(img_res)

        draw_img = drawSegLines(img, 7, box_new)
        # showImage(draw_img, pic_name)
        return [draw_img, box_new]

class NumAll(object):
    """docstring for NumAll"""
    def __init__(self, img, box_21, box_7, pic_name='0'):
        super(NumAll, self).__init__()
        self.data = self.numAll(img, box_21, box_7, pic_name)

    def numAll(self, img, box_21, box_7, pic_name):
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_img = drawSegLines(draw_img, 21, box_21)
        draw_img = drawSegLines(draw_img, 7, box_7)
        # showImage(draw_img)
        return draw_img
