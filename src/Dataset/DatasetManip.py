import json
import os

import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

EN_ROOT = '../dataset_v2/en/'
ZH_ROOT = '../dataset_v2/zh/'
EN_TFR_PATH = '../model/license_plate_en.tfrecords'
ZH_TFR_PATH = '../model/license_plate_zh.tfrecords'

STD_W = 45
STD_H = 140
STD_D = 1

en_classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35
}

zh_classes = {
    "zh_chuan": 0,
    "zh_e": 1,
    "zh_gan1": 2,
    "zh_gan4": 3,
    "zh_guiz": 4,
    "zh_guil": 5,
    "zh_hei": 6,
    "zh_hu": 7,
    "zh_ji2": 8,
    "zh_ji4": 9,
    "zh_jin1": 10,
    "zh_jin4": 11,
    "zh_jing": 12,
    "zh_liao": 13,
    "zh_lu": 14,
    "zh_meng": 15,
    "zh_min": 16,
    "zh_ning": 17,
    "zh_qing": 18,
    "zh_qiong": 19,
    "zh_shan": 20,
    "zh_su": 21,
    "zh_wan": 22,
    "zh_xiang": 23,
    "zh_xin": 24,
    "zh_yu2": 25,
    "zh_yu4": 26,
    "zh_yue": 27,
    "zh_yun": 28,
    "zh_zang": 29,
    "zh_zhe": 30
}


def _int64_feature(value):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def generate_dataset(type):

    picked_class = None
    picked_root = None
    tfr_path = None
    if type == 'en':
        picked_class = en_classes
        picked_root = EN_ROOT
        tfr_path = EN_TFR_PATH
    else:
        picked_class = zh_classes
        picked_root = ZH_ROOT
        tfr_path = ZH_TFR_PATH

    writer = tf.io.TFRecordWriter(tfr_path)

    label_key = 0
    for label, name in enumerate(picked_class):

        # print(f'label: {label} name: {name}')

        path = picked_root + str(name) + '/'

        for img_name in os.listdir(path):
            img_path = path + img_name

            original_img = Image.open(img_path)
            shape = [STD_H, STD_W, STD_D]

            img = original_img.convert('L').tobytes()

            features = {
                'height': _int64_feature(shape[0]),
                'width': _int64_feature(shape[1]),
                'depth': _int64_feature(shape[2]),
                'label': _int64_feature(label_key),
                'image_raw': _bytes_feature(img),
            }
            exmp = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(exmp.SerializeToString())
        label_key += 1
    writer.close()
    return tfr_path

img_features = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}


def parse_records(record):
    return tf.io.parse_single_example(record, features=img_features)


def parse_dataset(tfr_name):
    filenames = [tfr_name]

    reader = tf.data.TFRecordDataset(filenames)
    parsed_dataset = reader.map(parse_records)

    imgs = []
    labels = []

    for data in parsed_dataset:
        # print(f'{data["label"]}, {data["width"]}x{data["height"]}x{data["depth"]}')

        image_raw = data['image_raw'].numpy()

        restored_img = tf.io.decode_raw(image_raw, tf.uint8)
        restored_img = tf.reshape(restored_img, (data["height"], data["width"]))

        imgs.append(restored_img)
        labels.append(data['label'])

        # plt.imshow(restored_img, cmap='gray')
        # plt.show()

    np_imgs = np.array(imgs)
    np_labels = np.array(labels)

    return np_imgs, np_labels, [], []

    # train_imgs = []
    # train_lbls = []
    # test_imgs = []
    # test_lbls = []
    #
    # split_point = 1000
    # train_imgs = np_imgs[:split_point]
    # train_lbls = np_labels[:split_point]
    # test_imgs = np_imgs[split_point + 1:]
    # test_lbls = np_labels[split_point + 1:]

    # print(f'{train_imgs.shape} - {train_lbls.shape} | {test_imgs.shape} - {test_lbls.shape}')

    # return train_imgs, train_lbls, test_imgs, test_lbls
