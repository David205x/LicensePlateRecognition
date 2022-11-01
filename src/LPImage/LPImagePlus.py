import math
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_gray_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def preidentification(img_gray, img_HSV, img_B, img_R):
    img_h, img_w = img_HSV.shape[0:2]

    for i in range(img_h):
        for j in range(img_w):
            if ((img_HSV[:, :, 0][i, j] - 115) ** 2 < 0xe1) and (img_B[i, j] > 70) and (img_R[i, j] < 40):
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    # 定义核
    kernel_small = np.ones((3, 3))
    kernel_big = np.ones((7, 7))

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_di = cv2.dilate(img_gray, kernel_small, iterations=5)
    img_close = cv2.morphologyEx(img_di, cv2.MORPH_CLOSE, kernel_big)
    img_close = cv2.GaussianBlur(img_close, (5, 5), 0)
    _, img_bin = cv2.threshold(img_close, 115, 255, cv2.THRESH_BINARY)

    # show_gray_img(img_bin)

    return img_bin


def fix_position(img, img_bin):
    # 检测所有外轮廓，只留矩形的四个顶点
    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # 形状及大小筛选校验
    det_x_max = -1
    det_y_max = -1
    index = 0
    for i in range(len(contours)):
        x_min = np.min(contours[i][:, :, 0])
        x_max = np.max(contours[i][:, :, 0])
        y_min = np.min(contours[i][:, :, 1])
        y_max = np.max(contours[i][:, :, 1])
        det_x = x_max - x_min
        det_y = y_max - y_min
        if det_y == 0:
            return None
        ratio = det_x / det_y
        area = det_x * det_y
        if (ratio > 2.5) and (ratio < 5) and (det_x > det_x_max) and (det_y > det_y_max):
            det_y_max = det_y
            det_x_max = det_x
            index = i

    # 获取最可疑区域轮廓点集
    points = np.array(contours[index][:, 0])

    return points


def findVertices(points, offset=0):
    rect = cv2.minAreaRect(points)
    box = np.int0(cv2.boxPoints(cv2.minAreaRect(points)))

    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

    # 上下左右四个点坐标
    vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                         [right_point_x, right_point_y]])
    r = [vertices[0][0] - offset, vertices[0][1] - offset, vertices[1][0] + offset, vertices[1][1] + offset]

    if r[0] > r[2]:
        r[0], r[2] = r[2], r[0]
    if r[1] > r[3]:
        r[1], r[3] = r[3], r[1]

    return vertices, r


def extract_roi(img, rect, offset=0):
    x, y, w, h = int(rect[0] - offset), \
                 int(rect[1] - offset), \
                 int(rect[2] - rect[0] + 2 * offset), \
                 int(rect[3] - rect[1] + 2 * offset)

    return img[y: y + h, x: x + w].copy()


def cast_shadows(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, result = cv2.threshold(img_gray, 0x88, 0xff, cv2.THRESH_BINARY)
    height, width = result.shape
    dot = [0 for _ in range(width)]

    for j in range(width):
        for i in range(height):
            if result[i, j] == 0:
                dot[j] += 1
                result[i, j] = 0xff

    for j in range(width):
        for i in range((height - dot[j]), height):
            result[i, j] = 0

    dot_flipped = [height - dot[i] for i in range(width)]
    #
    # return cv2.flip(result, 0), dot_flipped

    return result, dot_flipped


def analyze_shadows(img, image_shd, dots):
    slices = []

    height, width = image_shd.shape[0:2]

    lowest = 0xffff
    avg_shd = sum_shd = 0
    for w in range(width):
        sum_shd += dots[w]
    avg_shd = int(sum_shd / width) >> 2

    cur = image_shd[avg_shd, 0]
    for w in range(width):
        if not image_shd[avg_shd, w] == cur:
            slices.append(w)
            cur = ((cur + 1) % 2) * 0xff

    dist = []
    for i in range(len(slices) - 1, 0, -1):
        dist.append(slices[i] - slices[i - 1])

    approx_width = int(width / 14)
    gaps = []
    chars = []
    valid_counter = 0

    for d in dist:
        if d > approx_width:
            chars.append(d)
            valid_counter += 1
        else:
            gaps.append(d)

        if valid_counter == 5:
            break
    print(f'{chars}\t{gaps}')

    gap_avg = 0
    char_avg = 0

    if len(chars) == 0:
        return None, None

    for c in chars:
        char_avg += c
    char_avg /= len(chars)

    # for g in gaps:
    #     gap_avg += g
    # gap_avg /= len(gaps)

    ratio_const = GAP_CHAR_RATIO_CONST
    gap_avg = char_avg * ratio_const

    print(f'({char_avg}\t{gap_avg}) ratio: {char_avg / gap_avg}')

    return round(char_avg) + EXTRA_CHAR_WIDTH, math.ceil(gap_avg)


def perspective_warp(img, rect, offset=0):
    dw = 440
    dh = 140

    rect = [rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset]

    src = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
    dst = [(0, 0), (dw, 0), (dw, dh), (0, dh)]

    warp_src, warp_dst = np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(warp_src, warp_dst)

    return cv2.warpPerspective(img, mat, dsize=(dw, dh))


GAP_CHAR_RATIO_CONST = 0.28
BIGGER_GAP_CHAR_RATIO_CONST = 0.8
EXTRA_CHAR_WIDTH = 1


def draw_analysis(img, char_width, gap_width):
    height, width = img.shape[0:2]

    half_char_width = char_width >> 1
    half_gap_width = gap_width >> 1

    ratio_const = BIGGER_GAP_CHAR_RATIO_CONST
    bigger_gap_width = int(char_width * ratio_const)

    anchors = []
    cur = 0
    anchors.append(cur)

    cur += gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += bigger_gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += gap_width
    anchors.append(cur)

    cur += char_width
    anchors.append(cur)
    cur += gap_width
    anchors.append(cur)

    print(anchors)

    visible = 0
    for i in range(len(anchors) - 1):
        if visible == 1:
            rect = [anchors[i], 0, anchors[i + 1], height]
            draw_rect(img, rect)
        visible = (visible + 1) % 2

    show_image(img)

    return anchors


def draw_rect(img, r, offset=0):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    cv2.rectangle(img,
                  (r[0] - offset, r[1] - offset),
                  (r[2] + offset, r[3] + offset),
                  color, 2)
    return


class LPImagePlus(object):
    img = None
    lp_img = None
    image_path = None
    processed_img = None
    shadow_details = []
    slices = []

    def __init__(self, path):
        self.image_path = path
        try:
            self.img = plt.imread(self.image_path)
        except Exception as e:
            print(e)
        finally:
            return

    def show_self(self, color_map):
        plt.imshow(self.img, cmap=color_map)
        plt.show()

    def preprocess(self):

        # 统一规定大小
        self.img = cv2.resize(self.img, (640, 480))
        img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 高斯模糊
        img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
        # RGB通道分离
        img_B = cv2.split(img_gauss)[0]
        img_G = cv2.split(img_gauss)[1]
        img_R = cv2.split(img_gauss)[2]
        # 读取灰度图和HSV空间图
        img_gray = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)
        img_HSV = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2HSV)

        return img_gray, img_HSV, img_B, img_R

    def process(self):

        h, w = self.img.shape[0:2]

        if w == 440 and h == 140:
            rect = [0, 0, 440, 140]
        else:
            img_gray, img_HSV, img_B, img_R = self.preprocess()
            img_bin = preidentification(img_gray, img_HSV, img_B, img_R)
            pts = fix_position(self.img, img_bin)
            if pts is None:
                return
            _, rect = findVertices(pts, 0)

        self.lp_img = perspective_warp(self.img, rect, -4)
        roi_img = self.lp_img

        img_shadows, dts = cast_shadows(roi_img)

        char_width, gap_width = analyze_shadows(roi_img, img_shadows, dts)

        if char_width is not None and gap_width is not None:
            draw_analysis(roi_img, char_width, gap_width)

        # draw_rect(self.img, rect)
        # show_image(self.img)
