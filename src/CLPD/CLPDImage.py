import math
import random
import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


class CLPDImage(object):

    img = None
    lp_img = None

    img_path = None
    rect = []

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