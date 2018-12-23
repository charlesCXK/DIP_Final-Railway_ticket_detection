# --*-- encoding: utf-8 --*--
import cv2

# read and binarize the pictures of train_data
class Binarization(object):
	"""docstring for Binarization"""
	def __init__(self, train_data):
		super(Binarization, self).__init__()
		self.train_data = train_data
		self.bin_img_list = self.binarize(train_data)

	def binarize(selfself, train_data):
		bin_img = []
		for lines in train_data:
			pic_path = '../train_data/' + lines
			# flag = os.path.exists(pic_path)
			# abspath = os.path.abspath(pic_path)
			pic = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
			# pic = cv2.adaptiveThreshold(pic, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
			ret, pic = cv2.threshold(pic, 100, 255, cv2.THRESH_BINARY)
			# print(pic_path)
			# cv2.namedWindow("1", 0)
			# cv2.resizeWindow("1", 1200, 900)
			# cv2.imshow('1', pic)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			bin_img.append(pic)
		return bin_img
