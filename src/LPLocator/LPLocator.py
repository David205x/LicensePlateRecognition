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


def crop_rect(img, rect, offset=0):
    x, y, w, h = int(rect[0] - offset), \
                 int(rect[1]), \
                 int(rect[2] - rect[0] + 2 * offset), \
                 int(rect[3] - rect[1])

    cropped = img[y: y + h, x: x + w].copy()
    return cropped


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

    # show_gray_img(img_bin)

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

        if (ratio > 3) and (ratio < 4) and (w > w_max) and (h > h_max):
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
    ly = np.min(box[:, 1][np.where(box[:, 0] == lx)])

    rx = np.max(box[:, 0])
    ry = np.max(box[:, 1][np.where(box[:, 0] == rx)])

    ty = np.min(box[:, 1])
    tx = np.min(box[:, 0][np.where(box[:, 1] == ty)])

    by = np.max(box[:, 1])
    bx = np.max(box[:, 0][np.where(box[:, 1] == by)])

    # print(box)
    # print(f'\t\t({tx}, {ty})')
    # print(f'({lx}, {ly})\t\t({rx}, {ry})')
    # print(f'\t\t({bx}, {by})')

    v = []
    if ang >= 45:
        v = np.array([(lx, ly), (tx, ty), (rx, ry), (bx, by)])
    else:
        v = np.array([(tx, ty), (rx, ry), (bx, by), (lx, ly)])

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

    # print(f'{vertices} -> {dst}')

    warp_src, warp_dst = np.float32(src), np.float32(dst)
    # np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

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


def cast_shadows(lp_img):
    img_gray = cv2.cvtColor(lp_img, cv2.COLOR_RGB2GRAY)
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
    return result, dot_flipped


def analyze_shadows(image_shd, dots):
    height, width = image_shd.shape[0:2]
    blocks_pos = []

    sum_shd = 0

    for w in range(width):
        sum_shd += dots[w]
    reference_height = (int(sum_shd / width) >> 3) + 2
    # print(f'sampling at h={reference_height}')

    block_type = [0, 0xff]

    first_block_type = cur_type = image_shd[reference_height, 0]
    blocks_pos.append(0)
    for w in range(width):
        if not image_shd[reference_height, w] == cur_type:
            blocks_pos.append(w)
            cur_type = block_type[((cur_type + 1) % 2)]
    blocks_pos.append(width)
    last_block_type = cur_type

    dist = []
    for i in range(0, len(blocks_pos) - 1):
        dist.append(blocks_pos[i + 1] - blocks_pos[i])
    rev_dist = dist
    rev_dist.reverse()

    min_char_width = int(width / 25)
    noise_width = int(width / 100)
    approx_char_max = width
    approx_char_min = int(width / 14)
    wake_fixer = 0.6 * STD_CHAR_WIDTH
    fixer_const = 0.33 * STD_CHAR_WIDTH
    gaps = []
    chars = []
    char_counter = 0

    right_start = 0
    fixer_called = False

    if len(rev_dist) < 4:
        return None, None, None

    # print(rev_dist)

    if last_block_type == block_type[0]:  # last block is black
        # print(f'Applying case 0x00')
        if rev_dist[0] > wake_fixer and rev_dist[2] > wake_fixer:
            right_start -= fixer_const
        skipped = False
        for d in rev_dist:
            if not skipped:  # skip the first block
                skipped = True
                right_start += d
                continue
            if approx_char_max > d > min_char_width:
                chars.append(d)
                char_counter += 1
            elif d > approx_char_max:
                continue
            else:
                gaps.append(d)
            if char_counter == 5:
                break
    else:  # last block is white
        # print(f'Applying case 0xff')
        if rev_dist[1] > wake_fixer and rev_dist[3] > wake_fixer:
            right_start -= fixer_const
        processing_first = True
        for d in rev_dist:
            if processing_first:  # check whether the last block has a suitable width for a character
                if noise_width < d < approx_char_min:
                    right_start += d
                else:
                    processing_first = False
                continue
            if approx_char_max > d > min_char_width:
                chars.append(d)
                char_counter += 1
            elif d > approx_char_max:
                continue
            else:
                gaps.append(d)
            if char_counter == 5:
                break

    # print(f'{chars} | {gaps}')

    char_avg = 0

    if len(chars) == 0:
        return None, None, None

    for c in chars:
        char_avg += c
    char_avg /= len(chars)

    char_avg = (char_avg + 135) / 4
    gap_avg = char_avg * GAP_CHAR_RATIO_CONST

    # print(f'{right_start}, ({char_avg}\t{gap_avg}) ratio: {char_avg / gap_avg}')

    return int(right_start), round(char_avg) + EXTRA_CHAR_WIDTH, math.ceil(gap_avg)


def crop_chars(lp_img, right_start, char_width, gap_width, std=False):
    height, width = lp_img.shape[0:2]

    lp_gray_img = cv2.cvtColor(lp_img, cv2.COLOR_RGB2GRAY)

    bigger_gap_width = int(char_width * BIGGER_GAP_CHAR_RATIO_CONST)

    width_preset = [
        [
            right_start,
            char_width, gap_width,
            char_width, gap_width,
            char_width, gap_width,
            char_width, gap_width,
            char_width
        ],
        [
            gap_width,
            char_width, gap_width,
            char_width, bigger_gap_width
        ]
    ]

    std_width_preset = [
        [
            12,
            45, 12,
            45, 12,
            45, 12,
            45, 12,
            45],
        [
            12,
            45, 12,
            45, 34
        ]
    ]

    picked_preset = []
    if std:
        print(f'Slicing with standard preset...')
        picked_preset = std_width_preset
    else:
        print(f'Slicing with customized preset...')
        picked_preset = width_preset

    picked_preset = std_width_preset

    r_cur = width
    l_cur = 0
    r_anchors = []
    l_anchors = []
    for w in picked_preset[0]:
        r_anchors.append(r_cur - w)
        r_cur -= w
    for w in picked_preset[1]:
        l_anchors.append(l_cur + w)
        l_cur += w

    sliced_digits = []
    sliced_chars = []

    r_visible = 1
    r_success_count = 0
    for i in range(len(r_anchors) - 1):
        if r_visible == 1:
            rect = [r_anchors[i + 1], 0, r_anchors[i], height - 0]
            offset = 6
            # draw_rect(self.lp_img, rect, offset)
            tmp = crop_rect(lp_gray_img, rect, offset)
            r_success_count += 1
            if r_success_count == 6:
                break
            try:
                tmp = cv2.resize(tmp, (45, 140))
            except Exception as e:
                continue
            # show_gray_img(tmp)
            sliced_digits.append(tmp)
        r_visible = (r_visible + 1) % 2

    l_visible = 1
    l_success_count = 0
    for i in range(len(l_anchors) - 1):
        if l_visible == 1:
            rect = [l_anchors[i], 0, l_anchors[i + 1], height - 0]
            offset = 6
            # draw_rect(self.lp_img, rect, offset)
            tmp = crop_rect(lp_gray_img, rect, offset)
            l_success_count += 1
            if l_success_count == 3:
                break
            try:
                tmp = cv2.resize(tmp, (45, 140))
            except Exception as e:
                continue
            # show_gray_img(tmp)
            sliced_chars.append(tmp)
        l_visible = (l_visible + 1) % 2

    # show_image(self.lp_img)

    sliced_imgs = []

    for sc in sliced_chars:
        sliced_imgs.append(sc)

    sliced_digits.reverse()
    for sd in sliced_digits:
        sliced_imgs.append(sd)

    return sliced_imgs


LP_WIDTH = 440
LP_HEIGHT = 140

STD_CHAR_WIDTH = 45

# 0.267 0.756 0
GAP_CHAR_RATIO_CONST = 0.3
BIGGER_GAP_CHAR_RATIO_CONST = 0.7
EXTRA_CHAR_WIDTH = 2


class LPLocator(object):

    def __init__(self, path):

        self.image_path = path

        self.img = None
        self.lp_img = None

        self.w = None
        self.h = None

        self.std_flag = False

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

    def rough_process(self):

        rect = []
        vertices = []
        dots = []
        slices = []

        if self.w == 440 and self.h == 140:
            rect = [0, 0, 440, 140]
            vertices = [
                (rect[0], rect[1]),
                (rect[2], rect[1]),
                (rect[2], rect[3]),
                (rect[0], rect[3])
            ]
            self.std_flag = True

        else:
            img_gray, img_HSV, img_B, img_R = self.preprocess()
            img_bin = preprocess(img_gray, img_HSV, img_B, img_R)
            pts = find_lp(img_bin)

            if pts is None:
                return self.img, self.lp_img, []

            vertices, rect = locate_rect(pts, -6)

        self.lp_img = perspective_warp(self.img, vertices)
        draw_rect(self.img, rect, vertices, 4)
        # show_image(self.img)
        lp_shadow_img, dots = cast_shadows(self.lp_img)
        right_start, char_w, gap_w = analyze_shadows(lp_shadow_img, dots)

        if char_w is not None and gap_w is not None:
            slices = crop_chars(self.lp_img, right_start, char_w, gap_w, self.std_flag)
            return self.img, self.lp_img, slices
        else:
            return self.img, self.lp_img, []
