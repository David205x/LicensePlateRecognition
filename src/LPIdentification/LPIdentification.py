import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.resnet import preprocess_input
from keras.utils import image_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow import keras

import cv2
import numpy as np

from matplotlib.pyplot import plot
import matplotlib.pylab as pylab

CLASSES_COUNT = 36
STD_W = 45
STD_H = 140
STD_D = 1

TEST_IMG_STORAGE = '../img/'

classes = {0: '0',
           1: '1',
           2: '2',
           3: '3',
           4: '4',
           5: '5',
           6: '6',
           7: '7',
           8: '8',
           9: '9',
           10: 'a',
           11: 'b',
           12: 'c',
           13: 'd',
           14: 'e',
           15: 'f',
           16: 'g',
           17: 'h',
           18: 'i',
           19: 'j',
           20: 'k',
           21: 'l',
           22: 'm',
           23: 'n',
           24: 'o',
           25: 'p',
           26: 'q',
           27: 'r',
           28: 's',
           29: 't',
           30: 'u',
           31: 'v',
           32: 'w',
           33: 'x',
           34: 'y',
           35: 'z'}


def predict(file, model):
    img = plt.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    target = tf.reshape(image_utils.img_to_array(img), (1, STD_H, STD_W))
    pred = np.argmax(model.predict(target))

    print(f'{file} -> {classes.get(pred)}')


def one_hot(labels):
    onehot_labels = np.zeros(shape=[len(labels), CLASSES_COUNT])
    for i in range(len(labels)):
        onehot_labels[i][labels[i]] = 1
    return onehot_labels


class LPIdentification(object):
    save_path = '../model/LPModel.h5'

    def __init__(self, tfrecord_path, train_images, train_labels):

        self.record_path = tfrecord_path
        self.model = None
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = train_images[-100:]
        self.test_labels = train_labels[-100:]

        print(f'{train_images.shape} : {train_labels.shape} | {self.test_images.shape} : {self.test_labels.shape} ')

    def mnist_cnn(self, _input_shape):

        model = keras.Sequential()

        model.add(Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu', input_shape=_input_shape))
        model.add(MaxPool2D(pool_size=2, strides=2, padding="valid"))
        model.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(CLASSES_COUNT, activation='softmax'))

        print(model.summary())
        self.model = model

    def train_model(self, _train_images, _train_labels, _test_images, _test_labels):

        _train_images = _train_images.reshape(-1, STD_H, STD_W, STD_D) / 255.0
        _test_images = _test_images.reshape(-1, STD_H, STD_W, STD_D) / 255.0

        _train_labels = one_hot(_train_labels)
        _test_labels = one_hot(_test_labels)

        self.mnist_cnn(_input_shape=(STD_H, STD_W, STD_D))
        self.model.compile(optimizer=tf.optimizers.Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
        self.model.fit(x=_train_images, y=_train_labels, epochs=25)

        loss, acc = self.model.evaluate(x=_test_images, y=_test_labels)
        metrics = [tf.keras.metrics.sparse_categorical_accuracy]

        print(f'accuracy: {acc}\nloss: {loss}\nmetrics: {metrics}')

        corr = 0
        predictions = self.model.predict(_test_images)
        for i in range(len(_test_images)):
            target = np.argmax(predictions[i])
            label = np.argmax(_test_labels[i])
            if label == target:
                corr += 1
        print(f'correct count: {corr}\ncorrect pct: {corr / len(_test_images)}')

        self.model.save(LPIdentification.save_path, overwrite=True)
        print(f'Model saved at {LPIdentification.save_path}')

    def execute_identification(self):
        # mnist = (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        self.train_model(self.train_images, self.train_labels, self.test_images, self.test_labels)

        h5_model = tf.keras.models.load_model(LPIdentification.save_path)
        if h5_model is None:
            return

        print('*' * 40)
        dir_list = os.listdir(TEST_IMG_STORAGE)
        dir_list.sort()

        print(dir_list)

        for f in dir_list:
            predict(TEST_IMG_STORAGE + f, h5_model)
