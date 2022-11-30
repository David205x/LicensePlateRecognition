import os
import math
import sys
import cv2

# from src.LPLocator.LPLocator import LPLocator
from CSVReader.CSVReader import CSVReader
from Dataset.DatasetManip import generate_records, parse_dataset
from LPIdentification.LPIdentification import LPIdentification
from LPLocator.LPLocator import LPLocator
import matplotlib.pyplot as plt

from src.RecordsMng.RecordsMng import RecordsMng

BASE_DATASET_PATH = '../CLPD_1200/'
LP_TEST_IMGS_PATH = '../lp/'

EN_TFR_PATH = '../model/license_plate_en.tfrecords'
ZH_TFR_PATH = '../model/license_plate_zh.tfrecords'


def result_coversion(result_arr):
    result_str = result_arr[0]
    for r in result_arr[1]:
        result_str += str(r)
    ret_str = ''
    ret_str += result_str[:2]
    ret_str += '·'
    ret_str += result_str[2:]

    return ret_str


records_manager = RecordsMng(gen_new=False, train_new=True)        # TODO UNTEST
en_identifier, zh_identifier = records_manager.get_identifiers()


def get_summary():
    print(records_manager.en_identifier.h5_model.summary())
    print(records_manager.en_identifier.h5_model.summary())


def Main(photo_path):

    print("|--License Plate Recognition--|")

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # desired_csv_size = 100
    # cr = CSVReader(desired_csv_size)

    # col_titles = cr.get_csv_title()
    # csv_data = cr.get_csv_data()
    # print(f'CSV data loaded.')

    # files = os.listdir(LP_TEST_IMGS_PATH)

    current_file = photo_path
    print(f'Loaded {current_file}.')
    # >>>>>>>>>>> 2022-11-14 模型展示
    get_summary()
    locator = LPLocator(current_file)
    img_lp_highlighted, img_lp_cropped, char_imgs = locator.rough_process()     # TODO crashed when no palette detected
    img, shadow_image, img_with_rects = locator.return_image()
    if len(char_imgs) == 0:
        print(f'Failed to identify the license...')
        final_result = '识别失败'
    else:
        result = zh_identifier.identify_chars([char_imgs[0]])
        result.append(en_identifier.identify_chars(char_imgs[1:]))
        final_result = result_coversion(result)

    return img_lp_highlighted, shadow_image, img_with_rects, final_result

    # plt.imshow(img_lp_highlighted)
    # plt.imshow(shadow_image)
    # plt.imshow(sliced_photos)
    # plt.title(result_coversion(result))
    # plt.show()

    # os.system("pause")

    # os.system("pause")

    # img = CLPDImage(current_file)
    #
    # img.get_csv_data(csv_data[img.get_id()])
    #
    # img.preprocess()
    # # img.write_all()
