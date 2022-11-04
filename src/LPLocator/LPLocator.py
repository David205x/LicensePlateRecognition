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


def preprocess(img_gray, img_HSV, img_B, img_R):
    img_h, img_w = img_HSV.shape[0:2]

    for i in range(img_h):
        for j in range(img_w):
            if ((img_HSV[:, :, 0][i, j] - 115) ** 2 < 0xe1) and (img_B[i, j] > 80) and (img_R[i, j] < 40):
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

    show_gray_img(img_bin)

    return img_bin


def find_lp(img_bin):

    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    w_max = -1
    h_max = -1
    index = 0

    for i in range(len(contours)):
        x_min = np.min(contours[i][:, :, 0])
        x_max = np.max(contours[i][:, :, 0])
        y_min = np.min(contours[i][:, :, 1])
        y_max = np.max(contours[i][:, :, 1])
        w = x_max - x_min
        h = y_max - y_min
        if h == 0:
            return None
        ratio = w / h
        area = w * h

        if (ratio > 3) and (ratio < 5) and (w > w_max) and (h > h_max):
            h_max = h
            w_max = w
            index = i

    points = np.array(contours[index][:, 0])

    return points


def locate_rect(points, offset=0):

    rect = cv2.minAreaRect(points)
    ang = rect[-1]
    box = np.int0(cv2.boxPoints(rect))

    lx = np.min(box[:, 0])
    ly = box[:, 1][np.where(box[:, 0] == lx)][0]

    rx = np.max(box[:, 0])
    ry = box[:, 1][np.where(box[:, 0] == rx)][0]

    ty = np.min(box[:, 1])
    tx = box[:, 0][np.where(box[:, 1] == ty)][0]

    by = np.max(box[:, 1])
    bx = box[:, 0][np.where(box[:, 1] == by)][0]

    # print(box)
    # print(f'\t\t({tx}, {ty})')
    # print(f'({lx}, {ly})\t\t({rx}, {ry})')
    # print(f'\t\t({bx}, {by})')

    v = []
    if ang >= 45:
        v = np.array([[lx, ly], [tx, ty], [rx, ry], [bx, by]])
    else:
        v = np.array([[tx, ty], [rx, ry], [bx, by], [lx, ly]])

    r = [v[0][0] - offset, v[0][1] - offset, v[2][0] + offset, v[2][1] + offset]

    if r[0] > r[2]:
        r[0], r[2] = r[2], r[0]
    if r[1] > r[3]:
        r[1], r[3] = r[3], r[1]

    return v, r


def extract_lp(img, rect, offset=0):

    mh, mw = img.shape[0:2]

    x, y, w, h = int(rect[0] - offset) if rect[0] > offset else 0, \
                 int(rect[1] - offset) if rect[1] > offset else 0, \
                 int(rect[2] - rect[0] + 2 * offset) if rect[2] - rect[0] + 2 * offset > mw else mw, \
                 int(rect[3] - rect[1] + 2 * offset) if rect[3] - rect[1] + 2 * offset > mh else mh

    return img[y: y + h, x: x + w].copy()


def perspective_warp(img, vertices):

    dw = LP_WIDTH
    dh = LP_HEIGHT

    src = vertices
    dst = [(0, 0), (dw, 0), (dw, dh), (0, dh)]

    warp_src, warp_dst = np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(warp_src, warp_dst)

    return cv2.warpPerspective(img, mat, dsize=(dw, dh))


def draw_rect(img, r, v, stroke_width, offset=0):
    R, G, B = random.randint(192, 256), random.randint(192, 256), random.randint(192, 256)
    color = (R, G, B)
    icolor = (0xff - R, 0xff - G, 0xff - B)

    cv2.rectangle(img,
                  (r[0] - offset, r[1] - offset),
                  (r[2] + offset, r[3] + offset),
                  color, stroke_width)
    return


class LPLocator(object):

    def __init__(self, path):

        self.image_path = path

        self.img = None
        self.lp_img = None

        self.w = None
        self.h = None

        try:
            self.img = plt.imread(self.image_path)
            self.h, self.w = self.img.shape[0:2]

        except Exception as e:
            print(e)
        finally:
            return

    def show_self(self, color_map):
        plt.imshow(self.img, cmap=color_map)
        plt.show()

    def preprocess(self):

        self.img = cv2.resize(self.img, (640, 480))

        img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_gauss = cv2.GaussianBlur(img, (5, 5), 0)

        img_B = cv2.split(img_gauss)[0]
        img_G = cv2.split(img_gauss)[1]
        img_R = cv2.split(img_gauss)[2]

        img_gray = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)
        img_HSV = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2HSV)

        return img_gray, img_HSV, img_B, img_R

    def process(self):

        rect = []
        vertices = []

        if self.w == 440 and self.h == 140:
            rect = [0, 0, 440, 140]
            vertices = rect

        else:
            img_gray, img_HSV, img_B, img_R = self.preprocess()
            img_bin = preprocess(img_gray, img_HSV, img_B, img_R)
            pts = find_lp(img_bin)

            if pts is None:
                return

            vertices, rect = locate_rect(pts, 0)

        # self.lp_img = perspective_warp(self.img, vertices)
        # show_image(self.lp_img)

        draw_rect(self.img, rect, vertices, 3)
        show_image(self.img)


LP_WIDTH = 400
LP_HEIGHT = 140
