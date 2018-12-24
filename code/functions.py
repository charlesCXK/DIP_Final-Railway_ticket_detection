# --*-- encoding: utf-8 --*--
import cv2
import os
import numpy


# read and binarize the pictures of train_data
class Binarization(object):
    """docstring for Binarization"""

    def __init__(self, pic_ori):
        super(Binarization, self).__init__()
        self.pic_ori = pic_ori
        self.pic_bin = self.binarize(pic_ori)

    def binarize(self, pic_ori):
        bin_img = []
        for pic in pic_ori:
            # pic = cv2.adaptiveThreshold(pic, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
            ret, pic = cv2.threshold(pic, 100, 255, cv2.THRESH_BINARY)
            # print(lines[0])
            # cv2.namedWindow("1", 0)
            # cv2.resizeWindow("1", 1200, 900)
            # cv2.imshow('1', pic)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            bin_img.append(pic)
        return bin_img


def writeImg(pic_list, pic_name, dir):
    for i in range(len(pic_list)):
        # path = os.path.join(dir, pic_name[i])
        # print(path)
        if (os.path.exists(dir) == 0):
            os.mkdir(dir)
        cv2.imwrite(os.path.join(dir, pic_name[i]), pic_list[i])


# remove the zero padding on the left and right
def removePadding(pic_list):
    pic_res = []
    for pic in pic_list:
        pic_median = cv2.medianBlur(pic, 5)
        cols = len(pic[0])
        lines = int(numpy.size(pic) / cols)
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
        pic_res.append(pic)
    return pic_res


# # remove salt noise
# def removeSaltNoise(pic_bin):
#     pic_res = []
#     for pic in pic_bin:
#         pic_median = cv2.medianBlur(pic, 5)
#         cols = len(pic[0])
#         lines = int(numpy.size(pic) / cols)
#         for i in range(lines):
#             for j in range(cols):
#                 if pic_median[i][j] == 0:
#                     pic[i][j] = 0
#                     break
#         pic_res.append(pic)
#     return pic_res


def getRectangle(pic_list):
    pic_res = []
    # a white picture
    pic_blank = cv2.cvtColor(pic_list[0], cv2.COLOR_GRAY2RGB)
    pic_blank[:] = 255
    for pic in pic_list:
        pic_median = cv2.medianBlur(pic, 5)
        _, contours, hierarchy = cv2.findContours(pic_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # pic_RGB = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        # (B, G, R)
        draw_img0 = cv2.drawContours(pic_blank.copy(), contours, -1, (0, 0, 0), 2)
        draw_img0 = cv2.cvtColor(draw_img0, cv2.COLOR_RGB2GRAY)
        # cv2.imwrite('../draw_picture/' + str(i) + '.jpg', draw_img0)
        #
        # cv2.namedWindow("draw_img0", 0)
        # cv2.resizeWindow("draw_img0", 640, 480)
        # cv2.imshow('draw_img0', draw_img0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pic_res.append(draw_img0)
    return pic_res

def getLines(pic_list):
    pic_res = []
    for pic in pic_list:
        # canny = cv2.Canny(pic, 240, 255)
        # cv2.namedWindow("canny", 0)
        # cv2.resizeWindow("canny", 640, 480)
        # cv2.imshow('canny', canny)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pic_res.append(canny)
        lines = cv2.HoughLinesP(pic, 1, numpy.pi / 180, 20, 20, 100)
        pic_blank = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        pic_blank[:] = 255
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(pic_blank, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.namedWindow("canny", 0)
        cv2.resizeWindow("canny", 640, 480)
        cv2.imshow('canny', pic_blank)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return pic_res