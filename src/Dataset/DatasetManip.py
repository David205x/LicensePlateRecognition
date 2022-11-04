import json
import os
from io import BytesIO

import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display

SLICES_ROOT = '../dataset_v2/'
TFR_PATH = '../model/license_plate_test.tfrecords'

classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z'}


def _int64_feature(value):
    if not isinstance(value,list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value,list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def generate_dataset(tfr_path):
    writer = tf.io.TFRecordWriter(tfr_path)

    label_key = 0
    for label, name in enumerate(classes):

        # print(f'label: {label} name: {name}')

        path = SLICES_ROOT + name + '/'
        for img_name in os.listdir(path):
            img_path = path + img_name

            original_img = Image.open(img_path)
            shape = [140, 45, 1]

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

    for data in parsed_dataset.take(20):

        print(f'{data["label"]}, {data["width"]}x{data["height"]}x{data["depth"]}')

        image_raw = data['image_raw']
        restored_img = tf.io.decode_raw(image_raw, tf.uint8)
        restored_img = tf.reshape(restored_img, (data["height"], data["width"], data["depth"]))

        # plt.imshow(restored_img, cmap='gray')
        # plt.show()
