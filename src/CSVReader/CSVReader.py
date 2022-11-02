import os
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt

CSV_PATH = '../CLPD_1200/CLPD.csv'


class CSVReader(object):
    data_size = -1
    csv_data = []
    col_title = None

    def __init__(self, desired_data_size=100):

        self.data_size = desired_data_size

        try:
            f = open(CSV_PATH)

            col_title = f.readline()
            self.col_title = col_title.strip().split(',')

            current_size = 0
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    new_data = line.strip().split(',')
                    new_data[0] = '../' + new_data[0]

                    self.csv_data.append(new_data)
                    current_size += 1

                    if current_size == self.data_size:
                        break

            print(f'{self.data_size} CSV data loaded.')

        except FileNotFoundError as e:
            print('Failed to load CSV data.')

    def get_csv_title(self):
        return self.col_title

    def get_csv_data(self):
        return self.csv_data
