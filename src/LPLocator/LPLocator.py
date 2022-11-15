import math
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist


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
    # cv2.imshow("img_bin", img_bin)
    # show_gray_img(img_bin)

    return img_bin


def find_lp(img_bin):
    # cv2.imshow("test", cv2.Canny(img_bin, 32, 68))
    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
        print(f'{ratio} {i}')
        if (w > w_max) and (h > h_max):
            h_max = h
            w_max = w
            index = i

    print(f'{index}')
    points = np.array(contours[index][:, 0])

    return points


def locate_convex(points):
    return cv2.convexHull(points)


def dis(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def point_cmp(p_a, p_b, center):

    if p_a[0] == 0 and p_b[0] == 0:
        return p_a[1] > p_b[1]
    a = np.array(p_a).tolist()
    b = np.array(p_b).tolist()
    a[0], a[1] = p_a[0], 2 * center[1] - p_a[1]
    b[0], b[1] = p_b[0], 2 * center[1] - p_b[1]
    det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])

    if det > 0:
        return True
    if det < 0:
        return False

    return True

    # d1 = p_a[0] + p_a[1]
    # d2 = p_b[0] + p_b[1]
    # if d1 == d2:
    #     return p_a[0] > p_b[0]
    # return d1 > d2


def clockwise_sort_points(points):
    center = []
    x = 0
    y = 0
    for i in range(len(points)):
        x = x + points[i][0][0]
        y = y + points[i][0][1]
    center.append(int(x / len(points)))
    center.append(int(y / len(points)))

    for i in range(0, len(points) - 1):
        for j in range(0, len(points) - i - 1):
            if points[j][0][0] + points[j][0][1] > points[j + 1][0][0] + points[j + 1][0][1]:
                tmp = points[j][0].tolist()
                points[j][0] = points[j + 1][0]
                points[j + 1][0] = np.array(tmp)

    for i in range(0, len(points) - 1):
        for j in range(0, len(points) - i - 1):
            if point_cmp(points[j][0], points[j + 1][0], center):
                tmp = points[j][0].tolist()
                points[j][0] = points[j + 1][0]
                points[j + 1][0] = np.array(tmp)

   # print(points)
    return points, center


def order_points(pts):
    return clockwise_sort_points(pts)


def vec_len(p1, p2):
    return math.sqrt(p1 * p1 + p2 * p2)


def check_valid(pts, center):
    points = np.array(pts).tolist()

    for i in range(len(points)):
        points[i][0][0], points[i][0][1] = points[i][0][0], 2 * center[1] - points[i][0][1]

    vec_top = [points[1][0][0] - points[0][0][0], points[1][0][1] - points[0][0][1]]

    vec_bottom = [points[3][0][0] - points[2][0][0], points[3][0][1] - points[2][0][1]]

    vec_left = [points[0][0][0] - points[3][0][0], points[0][0][1] - points[3][0][1]]

    vec_right = [points[2][0][0] - points[1][0][0], points[2][0][1] - points[1][0][1]]

    # print(f'vec = {vec_top}\n{vec_bottom}\n{vec_left}\n{vec_right}')
    res = []
    if vec_top[0] == 0:
        vec_top[0] = vec_top[0] + 1
    if vec_top[1] == 0:
        vec_top[1] = vec_top[1] + 1

    if vec_bottom[0] == 0:
        vec_bottom[0] = vec_bottom[0] + 1
    if vec_bottom[1] == 0:
        vec_bottom[1] = vec_bottom[1] + 1

    if vec_left[0] == 0:
        vec_left[0] = vec_left[0] + 1
    if vec_left[1] == 0:
        vec_left[1] = vec_left[1] + 1

    if vec_right[0] == 0:
        vec_right[0] = vec_right[0] + 1
    if vec_right[1] == 0:
        vec_right[1] = vec_right[1] + 1

    res.append(abs(vec_bottom[0] / vec_top[0]))
    res.append(abs(vec_bottom[1] / vec_top[1]))
    res.append(abs(vec_right[0] / vec_left[0]))
    res.append(abs(vec_right[1] / vec_left[1]))

    print(f'res = {res}')

    if (abs(res[0] - 1) > 0.2 or abs(res[1] - 1) > 0.2) and (abs(res[2] - 1) > 0.2 or abs(res[3] - 1) > 0.2):
        return False

    return True


def find_four_points(contour_points):
    try:
        ret = cv2.approxPolyDP(np.array(contour_points), 7, 1)
        # print(ret.shape)
        if len(ret) == 4:
            ret, center = order_points(ret)
            ret = ret.tolist()
            min_sum = 0
            for i in range(4):
                if ret[i][0][0] + ret[i][0][1] < ret[min_sum][0][0] + ret[min_sum][0][1]:
                    min_sum = i
            ret[0], ret[1], ret[2], ret[3] = ret[min_sum], ret[(min_sum + 1) % 4], ret[(min_sum + 2) % 4], ret[
                (min_sum + 3) % 4]
            if check_valid(ret, center):
                return ret
            else:
                print("点集有误")
                return None
    except Exception as e:
        print(e)
    return None


def locate_rect(points, offset=0):
    rect = cv2.minAreaRect(points)
    try:

        ang = rect[-1]

        box = np.int0(cv2.boxPoints(rect))
        # print(box)
        lx = np.min(box[:, 0])
        ly = np.min(box[:, 1][np.where(abs(box[:, 0] - lx) <= 0)])

        rx = np.max(box[:, 0])

        ry = np.max(box[:, 1][np.where(abs(box[:, 0] - rx) <= 0)])

        ty = np.min(box[:, 1])
        # >>>>>>>>> 2022-11-14 before change
        # tx = np.min(box[:, 0][np.where(box[:, 1] == ty)])
        # >>>>>>>>> 2022-11-14 before change
        tx = np.max(box[:, 0][np.where(abs(box[:, 1] - ty) <= 0)])

        by = np.max(box[:, 1])
        # print(f'{[]} -> {by}')
        # >>>>>>>>> 2022-11-14 before change
        # bx = np.max(box[:, 0][np.where(box[:, 1] == by)])
        # >>>>>>>>> 2022-11-14 before change
        bx = np.min(box[:, 0][np.where(box[:, 1] == by)])
        # print(box)
        # print(f'\t\t({tx}, {ty})')
        # print(f'({lx}, {ly})\t\t({rx}, {ry})')
        # print(f'\t\t({bx}, {by})')

        v = []
        if abs(ang) >= 45:
            # >>>>>>>>> 2022-11-14 before change
            # v = np.array([(lx, ly), (tx, ty), (rx, ry), (bx, by)])
            # >>>>>>>>> 2022-11-14 before change
            v = np.array([(lx, ly), (tx, ty), (rx, ry), (bx, by)])
            # print(f'{[]} -> {v}')
        else:
            v = np.array([(tx, ty), (rx, ry), (bx, by), (lx, ly)])

        r = [v[0][0] - offset, v[0][1] - offset, v[2][0] + offset, v[2][1] + offset]
        return v, r
    except Exception as e:
        print(e)


def extract_lp(img, rect, offset=0):
    mh, mw = img.shape[0:2]

    x, y, w, h = int(rect[0] - offset) if rect[0] > offset else 0, \
                 int(rect[1] - offset) if rect[1] > offset else 0, \
                 int(rect[2] - rect[0] + 2 * offset) if rect[2] - rect[0] + 2 * offset > mw else mw, \
                 int(rect[3] - rect[1] + 2 * offset) if rect[3] - rect[1] + 2 * offset > mh else mh

    return img[y: y + h, x: x + w].copy()


def perspective_warp(img, vertices):
    dw = LP_WIDTH - 2
    dh = LP_HEIGHT - 2

    src = vertices
    dst = [(0, 0), (dw, 0), (dw, dh), (0, dh)]
    # print(f'{vertices} -> {dst}')

    warp_src, warp_dst = np.float32(src), np.float32(dst)
    # np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(warp_src, warp_dst)
    return cv2.warpPerspective(img, mat, dsize=(dw, dh))


def draw_lp(img, r, v, stroke_width, offset=0):
    R, G, B = random.randint(192, 256), random.randint(192, 256), random.randint(192, 256)
    color = (R, G, B)
    icolor = (0xff - R, 0xff - G, 0xff - B)

    cv2.line(img, v[0], v[1], color, stroke_width)
    cv2.line(img, v[1], v[2], color, stroke_width)
    cv2.line(img, v[2], v[3], color, stroke_width)
    cv2.line(img, v[3], v[0], color, stroke_width)
    return


def draw_rect(img, r, v, stroke_width, offset=0):
    R, G, B = random.randint(192, 256), random.randint(192, 256), random.randint(192, 256)
    color = (R, G, B)
    icolor = (0xff - R, 0xff - G, 0xff - B)

    cv2.rectangle(img,
                  (r[0] - offset, r[1] - offset),
                  (r[2] + offset, r[3] + offset),
                  color, stroke_width)
    return


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src, seq):
    """图像亮度增强"""
    f = False
    src = np.array(src)
    src.flags.writeable = True
    ret = get_lightness(src) / 140

    if get_lightness(src) > 140:
        f = True
        ret = get_lightness(src) / 140 - 0.68
        print("第" + str(seq) + "张，图片亮度足够，减弱" + str(ret))
    else:
        print("第" + str(seq) + "张，图片亮度不足，增强" + str(ret))

    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel
    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)

    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    fI = out / 255.0
    gamma = 1 + ret + 0.08 * ret / 0.6
    if not f:
        # 伽马变换
        gamma = 0.3 + ret
        # print("第" + str(seq) + "张" + str(1.8 * ret))
        # print("第" + str(seq) + "张" + str(gamma))

    out = np.power(fI, gamma)

    return out, ret


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


def cast_shadows(lp_img):

    lp_test, ret = aug(lp_img, "1")

    lp_test = lp_test * 255

    lp_test = lp_test.astype("uint8")

    img_gray = cv2.cvtColor(lp_test, cv2.COLOR_RGB2GRAY)

    # print(img_gray_test)

    # cv2.imshow(str(124), img_gray)
    # >>>>>>>>>> 2022-11-14 before change
    # img_gray = cv2.cvtColor(lp_img, cv2.COLOR_RGB2GRAY)
    # <<<<<<< 2022-11-14 before change

    _, result = cv2.threshold(img_gray, 0x8f, 0xff, cv2.THRESH_BINARY)

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
            draw_rect(lp_img, rect, 4, 4)
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
            # draw_rect(lp_img, rect, offset)
            draw_rect(lp_img, rect, 4, 4)
            tmp = crop_rect(lp_gray_img, rect, offset)
            l_success_count += 1
            # show_gray_img(tmp)  # 车牌前2字
            if l_success_count == 3:
                break
            try:
                tmp = cv2.resize(tmp, (45, 140))
            except Exception as e:
                continue
            #
            sliced_chars.append(tmp)
        l_visible = (l_visible + 1) % 2
    # show_gray_img(lp_img)
    # show_image(lp_img)  #定位

    sliced_imgs = []

    for sc in sliced_chars:
        sliced_imgs.append(sc)

    sliced_digits.reverse()
    for sd in sliced_digits:
        sliced_imgs.append(sd)

    return sliced_imgs, lp_img


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
        self.lp_img_with_rects = None
        self.shadow_image = None
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

    def return_image(self):

        print(f'{self.img is None} {self.shadow_image is None} {self.lp_img_with_rects is None}')

        if self.lp_img_with_rects is None:
            return self.img, self.shadow_image, self.lp_img
        else:
            return self.img, self.shadow_image, self.lp_img_with_rects

    def show_self(self, color_map):
        plt.imshow(self.img, cmap=color_map)
        plt.show()

    def preprocess(self):

        # self.img = cv2.resize(self.img, (640, 480))

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
        hull = []
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
                self.lp_img = self.img
                self.shadow_image, _ = cast_shadows(self.lp_img)
                return self.img, self.lp_img, []
            hull = locate_convex(pts)
            vertices, rect = locate_rect(pts, 5)
        res = []
        for i in range(4):
            res.append([np.array(vertices[i])])
        # print(np.array(res).shape)

        ret = find_four_points(hull)
        if ret is not None:
            vertices = ret
            self.lp_img = perspective_warp(self.img, vertices)
            cv2.polylines(self.img, [np.array(ret)], True, (random.randint(120, 250), random.randint(120, 250), random.randint(120, 250)), 3)
        else:
            self.lp_img = perspective_warp(self.img, vertices)
            draw_lp(self.img, rect, vertices, 3)
        # draw_rect(self.img, rect, vertices, 2)
        lp_shadow_img, dots = cast_shadows(self.lp_img)

        right_start, char_w, gap_w = analyze_shadows(lp_shadow_img, dots)
        self.shadow_image = lp_shadow_img
        # show_image(lp_shadow_img) #垂直投影

        if char_w is not None and gap_w is not None:
            slices, lp_img = crop_chars(self.lp_img, right_start, char_w, gap_w, self.std_flag)
            self.lp_img_with_rects = lp_img
            return self.img, self.lp_img, slices
        else:
            return self.img, self.lp_img, []
