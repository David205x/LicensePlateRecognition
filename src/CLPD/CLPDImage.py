import math
import random
import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

SLICES_STORAGE = '../slices/'


def show_image(img):
    plt.imshow(img)
    plt.show()
    return


def show_gray_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    return


def draw_rect(img, rect, offset=0):
    color = (random.randint(127, 255), random.randint(127, 255), random.randint(127, 255))

    cv2.rectangle(img,
                  (rect[0] - offset, rect[1] - offset),
                  (rect[2] + offset, rect[3] + offset),
                  color, 4)
    return


def crop_rect(img, rect, offset=0):
    x, y, w, h = int(rect[0] - offset), \
                 int(rect[1]), \
                 int(rect[2] - rect[0] + 2 * offset), \
                 int(rect[3] - rect[1])

    cropped = img[y: y + h, x: x + w].copy()
    return cropped


def analyze_shadows(image_shd, dots):
    slices = []
    left_start = -1

    height, width = image_shd.shape[0:2]

    lowest = 0xffff
    avg_shd = sum_shd = 0

    # Determine the line of pixels that will be used for char detection.
    for w in range(width):
        sum_shd += dots[w]
    avg_shd = int(sum_shd / width) - 2

    # Slice the row of pixels into blocks by detecting color shifts between pixels.
    cur = image_shd[avg_shd, 0]
    for w in range(width):
        if not image_shd[avg_shd, w] == cur:
            slices.append(w)
            cur = ((cur + 1) % 2) * 0xff

    # Get block length from right to left
    dist = []
    for i in range(len(slices) - 1, 0, -1):
        dist.append(slices[i] - slices[i - 1])

    # print(dist)

    approx_width = int(width / 16)
    gaps = []
    chars = []
    valid_counter = 0

    # Using the first 5 characters on the right to calcuate the average char width.
    for d in dist:
        if d > approx_width:
            chars.append(d)
            valid_counter += 1
            if valid_counter == 5:
                break
        else:
            gaps.append(d)

    # print(f'{chars} | {gaps}')

    char_avg = 0
    # gap_avg = 0

    if len(chars) == 0:
        return None, None, None

    for c in chars:
        char_avg += c
    char_avg /= len(chars)

    # Calibrating the char width by putting 0.5 weight on standard width and measured width.
    char_avg = (char_avg + 45.0) / 2 + 1

    # for g in gaps:
    #     gap_avg += g
    # gap_avg /= len(gaps)

    # Calibrating the gap width.
    ratio_const = GAP_CHAR_RATIO_CONST
    gap_avg = int((char_avg * ratio_const + 15) / 2) + 2

    # TODO: FIX LEFT-START VALUE

    # print(f'{left_start}, ({char_avg}\t{gap_avg}) ratio: {char_avg / gap_avg}')

    return int(left_start), round(char_avg) + EXTRA_CHAR_WIDTH, math.ceil(gap_avg)


class CLPDImage(object):

    imgs_saved = 0

    def __init__(self, path):

        self.img = None
        self.lp_img = None
        self.sliced_imgs = []

        self.lp_string = None

        self.image_path = path

        self.rect = []
        self.normalized_rect = []

        try:
            self.img = plt.imread(self.image_path)
            path_var = self.image_path.split('/')

            file_data = (path_var[-1].split('.'))
            self.id = int(file_data[0])
            self.format = file_data[1]

        except Exception as e:
            print(e)
        finally:
            return

    def get_id(self):
        return self.id

    def get_csv_data(self, csv_data):

        self.lp_string = csv_data[-1]

        pts = []
        for i in range(1, 9):
            pts.append(int(csv_data[i]))

        for i in range(0, 8, 2):
            self.rect.append((pts[i], pts[i + 1]))

        # print(self.rect)
        return

    def show_original(self):
        plt.imshow(self.img)
        plt.show()

    def show_license(self):
        if self.lp_img is None:
            return
        plt.imshow(self.lp_img)
        plt.show()

    def show_gray_license(self):
        if self.lp_img is None:
            return
        plt.imshow(self.lp_img, cmap='gray')
        plt.show()

    def preprocess(self):

        print(f'Processing {self.image_path}...')
        self.perspective_warp(self.rect)

        image_shd, dots = self.cast_shadows()

        left_start, char_width, gap_width = analyze_shadows(image_shd, dots)

        if char_width is not None and gap_width is not None:
            self.draw_analysis(left_start, char_width, gap_width)

        return

    def perspective_warp(self, rect, offset=0):

        dw = LP_WIDTH
        dh = LP_HEIGHT

        src = rect
        dst = [(0, 0), (dw, 0), (dw, dh), (0, dh)]

        warp_src, warp_dst = np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

        mat = cv2.getPerspectiveTransform(warp_src, warp_dst)

        self.lp_img = cv2.warpPerspective(self.img, mat, dsize=(dw, dh))

        return

    def cast_shadows(self):

        img_gray = cv2.cvtColor(self.lp_img, cv2.COLOR_RGB2GRAY)
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

        # show_gray_image(result)

        # return cv2.flip(result, 0), dot_flipped
        return result, dot_flipped

    def draw_analysis(self, left_start, char_width, gap_width):

        height, width = self.lp_img.shape[0:2]

        half_char_width = char_width >> 1
        half_gap_width = gap_width >> 1

        ratio_const = BIGGER_GAP_CHAR_RATIO_CONST
        bigger_gap_width = int(char_width * ratio_const)

        anchors = []
        cur = half_gap_width + left_start
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

        visible = 0
        for i in range(len(anchors) - 1):
            if visible == 1:
                rect = [anchors[i], 0, anchors[i + 1], height]
                # draw_rect(self.lp_img, rect, 4)
                self.sliced_imgs.append(crop_rect(self.lp_img, rect, 4))
            visible = (visible + 1) % 2

        # print(anchors)

        return anchors

    def write_all(self):
        if len(self.sliced_imgs) == 0:
            print(f'{self.image_path} is not suitable for cropping.')
            return
        for img in self.sliced_imgs:
            try:
                writable = cv2.resize(img, (45, 140))
            except Exception as e:
                continue
            plt.imsave(SLICES_STORAGE + str(CLPDImage.imgs_saved) + '.jpg', writable)
            CLPDImage.imgs_saved += 1
        print(f'{self.image_path} cropped and saved.')

LP_WIDTH = 440
LP_HEIGHT = 140

# 0.267 0.756
GAP_CHAR_RATIO_CONST = 0.3
BIGGER_GAP_CHAR_RATIO_CONST = 0.82
EXTRA_CHAR_WIDTH = 2
