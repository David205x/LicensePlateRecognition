import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_CONTOUR_WIDTH = 80
MIN_CONTOUR_HEIGHT = 30

MIN_CONTOUR_RATIO = 1.5
MAX_CONTOUR_RATIO = 5

MIN_CONTOURL_AREA = 1500


def show_image(img):
    plt.imshow(img)
    plt.show()


def denoisilization(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    img_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    image_plus_black = cv2.add(image, img_blackhat)
    return cv2.subtract(image_plus_black, img_tophat)


def channel_extraction(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hue, image_saturation, image_value = cv2.split(image_hsv)
    return image_saturation


def find_rect(contour):
    y, x = [], []

    for c in contour:
        y.append(c[0][0])
        x.append(c[0][1])

    return [min(y), min(x), max(y), max(x)]


def license_location(img, original_image):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    possible_areas = []

    # TODO: 添加一个宽高比的判断，和颜色等权重


    for c in contours:
        rect = find_rect(c)
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])

        if rect[3] - rect[1] == 0:
            ratio = 0
        else:
            ratio = (rect[2] - rect[0]) / (rect[3] - rect[1])

        if ratio < 2 or ratio > 5 or area < 1440:
            continue

        possible_areas.append([rect, area, ratio])

    # 选取最蓝的部分作为匹配结果
    weight_max, index_max = -1, -1

    # h = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    # show_image(h)

    for i in range(len(possible_areas)):
        b = original_image[
            possible_areas[i][0][1]:possible_areas[i][0][3],
            possible_areas[i][0][0]:possible_areas[i][0][2]
            ]

        hsv = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
        lower = np.array([0x80, 0x70, 0x70])
        upper = np.array([0xa0, 0xff, 0xff])

        # 为所有备选区域绘制蒙版
        masks = cv2.inRange(hsv, lower, upper)
        weight_param1 = 0
        for m in masks:
            weight_param1 += m / 255

        weight_param2 = 0
        for w in weight_param1:
            weight_param2 += w

        if weight_param2 > weight_max:
            index_max = i
            weight_max = weight_param2

    # return possible_areas[index_max][0]
    return possible_areas

class LPImage(object):
    img = None
    image_path = None
    processed_img = None

    def __init__(self, path):
        self.image_path = path
        try:
            self.img = plt.imread(self.image_path)
            # tmp = cv2.resize(self.img, (400, 400 * self.img.shape[0] / self.img.shape[1]))
            # self.img = tmp
        except Exception as e:
            print(e)
        finally:
            return

    def show_self(self, color_map):
        plt.imshow(self.img, cmap=color_map)
        plt.show()

    def process(self):

        # 分离HSV通道，提取S（饱和度通道）作为基准图像
        image = channel_extraction(self.img)
        # 以3x3的矩形核进行礼帽-黑帽操作，去毛刺
        image = denoisilization(image)
        # 以OTSU法对图形进行阈值操作，完成二值化
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        threshold, image = cv2.threshold(gray, 0x80, 0xff, cv2.THRESH_OTSU)
        # 边缘检测
        canny_img = cv2.Canny(image, image.shape[0], image.shape[1])

        # 提取边缘预处理
        kernel = np.ones((5, 19), np.uint8)
        closing_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
        opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((11, 5), np.uint8)
        opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)

        rect = license_location(opening_img, self.img)

        for r in rect:
            cv2.rectangle(self.img, (r[0][0], r[0][1]), (r[0][2], r[0][3]), (255, 255, 0), 2)

        # cv2.rectangle(self.img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 0), 2)

        show_image(self.img)
