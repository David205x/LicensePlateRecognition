import os
import struct

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

CLASSES_COUNT = 68
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
           10: 'A',
           11: 'B',
           12: 'C',
           13: 'D',
           14: 'E',
           15: 'F',
           16: 'G',
           17: 'H',
           18: 'I',
           19: 'J',
           20: 'K',
           21: 'L',
           22: 'M',
           23: 'N',
           24: 'O',
           25: 'P',
           26: 'Q',
           27: 'R',
           28: 'S',
           29: 'T',
           30: 'U',
           31: 'V',
           32: 'W',
           33: 'X',
           34: 'Y',
           35: 'Z',
           36: "川",
           37: "鄂",
           38: "甘",
           39: "赣",
           40: "贵",
           41: "桂",
           42: "黑",
           43: "沪",
           44: "吉",
           45: "冀",
           46: "津",
           47: "晋",
           48: "京",
           49: "辽",
           50: "鲁",
           51: "蒙",
           52: "闽",
           53: "宁",
           54: "靑",
           55: "琼",
           56: "陕",
           57: "苏",
           58: "晋",
           59: "皖",
           60: "湘",
           61: "新",
           62: "豫",
           63: "渝",
           64: "粤",
           65: "云",
           66: "藏",
           67: "浙"
           }


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_gray_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def predict(file, model):
    img = plt.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    target = tf.reshape(image_utils.img_to_array(img_gray), (1, STD_H, STD_W))
    pred = np.argmax(model.predict(target))

    print(f'{file} -> {classes.get(pred)}')

    return True if file.split('/')[-1][0] == classes.get(pred) else False


def one_hot(labels):
    onehot_labels = np.zeros(shape=[len(labels), len(classes)])
    for i in range(len(labels)):
        onehot_labels[i][labels[i]] = 1
    return onehot_labels


def compute_matrix(TP, FP, TN, FN):  # TODO UNTEST
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return [accuracy, precision, recall, f1_score]


class LPIdentification(object):

    def __init__(self, tfrecord_path, train_images, train_labels, test_images, test_labels):

        self.record_path = tfrecord_path
        self.model_type = tfrecord_path.split('.')[-2][-2:]
        self.model_file_name = self.model_type + '_model.h5'
        self.save_path = '../model/' + self.model_file_name

        self.model = None
        self.h5_model = None
        self.train_images = train_images
        self.train_labels = train_labels

        # self.test_images = train_images[2::15]
        # self.test_labels = train_labels[2::15]

        self.test_images = test_images
        self.test_labels = test_labels

        # TODO UNTEST
        self.confusion = []
        self.evaluation = []

        print(f'{train_images.shape} : {train_labels.shape} | {self.test_images.shape} : {self.test_labels.shape} ')

    def build_model(self, _input_shape):

        model = keras.Sequential()

        model.add(Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu', input_shape=_input_shape))
        model.add(MaxPool2D(pool_size=4, strides=4, padding="valid"))
        model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPool2D(pool_size=2, strides=2, padding="valid"))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(CLASSES_COUNT, activation='softmax'))

        print(model.summary())
        self.model = model

    def train_model(self, _train_images, _train_labels, _test_images, _test_labels):

        _train_images = _train_images.reshape(-1, STD_H, STD_W, STD_D) / 255.0
        _test_images = _test_images.reshape(-1, STD_H, STD_W, STD_D) / 255.0

        _train_labels = one_hot(_train_labels)
        _test_labels = one_hot(_test_labels)

        self.build_model(_input_shape=(STD_H, STD_W, STD_D))

        if self.model_type == 'zh':
            epoch_base = 50
        else:
            epoch_base = 40

        learning_rates = [0.005]
        epoch_mul = [1]
        batch_sizes = [32]

        best_hyperparams = []
        cur_accuracy = -1.0

        for lr in learning_rates:
            self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy",
                               metrics=['accuracy'])
            for e in epoch_mul:
                for b in batch_sizes:
                    self.model.fit(verbose=0, x=_train_images, y=_train_labels, batch_size=b,
                                   epochs=int(epoch_base * e), shuffle=True)

                    loss, acc = self.model.evaluate(verbose=0, x=_test_images, y=_test_labels)

                    if acc > cur_accuracy:
                        cur_accuracy = acc
                        best_hyperparams = [lr, e, b]

                    metrics = [tf.keras.metrics.sparse_categorical_accuracy]
                    print('*' * 40)
                    print(
                        f'model type: {self.model_type}\nparam: l={lr}, e={int(epoch_base * e)}, b={b}\n'
                        f'accuracy: {acc}\nloss: {loss}\nmetrics: {metrics}')

                    # TODO UNTEST
                    conf_matrix = [[0 for j in range(68)] for i in range(68)]
                    # conf_matrix = [[0] * 68] * 68
                    eval_indicator = []
                    TP = [0] * 68
                    FP = [0] * 68
                    TN = [0] * 68
                    FN = [0] * 68

                    corr = 0
                    predictions = self.model.predict(_test_images)
                    for i in range(len(_test_images)):
                        target = np.argmax(predictions[i])
                        label = np.argmax(_test_labels[i])

                        conf_matrix[label][target] += 1
                        if label == target:
                            corr += 1
                            if self.model_type == 'en':  # TODO UNTEST
                                for ii in range(36):
                                    if ii == label:
                                        TP[label] += 1
                                    else:
                                        TN[label] += 1
                        elif self.model_type == 'en':
                            for ii in range(36):
                                if ii == label:
                                    FP[label] += 1
                                else:
                                    FN[label] += 1

                    # TODO UNTEST
                    if self.model_type == 'en':
                        for i in range(36):
                            eval_indicator.append(compute_matrix(TP[i], FP[i], TN[i], FN[i]))
                        self.evaluation = eval_indicator
                        self.confusion = conf_matrix
                        self.save_matrix(eval_indicator, 'evaluation.dat')
                        self.save_matrix(conf_matrix, 'confusion.dat')

                        # print(conf_matrix)
                        print(f'correct count: {corr}\ncorrect pct: {corr / len(_test_images)}')

                    print('*' * 40)

        self.model.save(self.save_path, overwrite=True)
        print(f'Best Hyperparameters: {best_hyperparams}')

        print(f'Model saved at {self.save_path}')

    def save_matrix(self, matrix, filename):
        with open('../model/' + filename, 'wb') as outfile:
            outfile.write(struct.pack('d', len(matrix[0])))
            outfile.write(struct.pack('d', len(matrix)))
            for i in matrix:
                for j in i:
                    outfile.write(struct.pack('d', j))

    def load_matrix(self, filename):
        matrix = []
        with open('../model/' + filename, 'rb') as readfile:
            column = struct.unpack('d', readfile.read(8))[0]
            row = struct.unpack('d', readfile.read(8))[0]
            print(column, ',', row)
            for i in range(int(row)):
                matrix.append(list(struct.unpack('{}d'.format(int(column)), readfile.read(8 * int(column)))))
                # line = []
                # for j in range(int(column)):
                #     line.append(struct.unpack('d', readfile.read(8))[0])
                # matrix.append(line)
        return matrix

    def load_h5_model(self, train_new):

        if train_new:
            self.train_model(self.train_images, self.train_labels, self.test_images, self.test_labels)
        self.h5_model = tf.keras.models.load_model(self.save_path)

        print(self.h5_model.summary())

        self.confusion = self.load_matrix('confusion.dat')
        self.evaluation = self.load_matrix('evaluation.dat')

    def identify_chars(self, char_imgs):
        result = []
        for cimg in char_imgs:
            target = tf.reshape(image_utils.img_to_array(cimg), (1, STD_H, STD_W))
            pred = np.argmax(self.h5_model.predict(target))

            mapped_result = classes.get(pred if self.model_type == 'en' else pred + 36)
            result.append(mapped_result)

        return result

    def training_test(self):
        if self.model is None:
            return

        print('*' * 40)
        dir_list = os.listdir(TEST_IMG_STORAGE)
        dir_list.sort()

        tot = len(dir_list)
        corr = 0

        for f in dir_list:
            if predict(TEST_IMG_STORAGE + f, self.h5_model):
                corr += 1
        print(f'Accuracy: {int(corr * 100 / tot)}%')
