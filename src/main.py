import os
import math

import cv2

# from src.LPLocator.LPLocator import LPLocator
from CSVReader.CSVReader import CSVReader
from Dataset.DatasetManip import generate_records, parse_dataset
from LPIdentification.LPIdentification import LPIdentification
from LPLocator.LPLocator import LPLocator
import matplotlib.pyplot as plt

BASE_DATASET_PATH = '../CLPD_1200/'
LP_TEST_IMGS_PATH = '../lp/'


def result_coversion(result_arr):
    result_str = result_arr[0]
    for r in result_arr[1]:
        result_str += str(r)
    ret_str = ''
    ret_str += result_str[:2]
    ret_str += '·'
    ret_str += result_str[2:]

    return ret_str


def Main(photo_path):

    print("|--License Plate Recognition--|")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # desired_csv_size = 100
    # cr = CSVReader(desired_csv_size)

    # col_titles = cr.get_csv_title()
    # csv_data = cr.get_csv_data()
    # print(f'CSV data loaded.')

    en_trf_path = generate_records('en')
    zh_trf_path = generate_records('zh')
    print(f'TFRecords generated.')

    en_train_images, en_train_labels, en_test_images, en_test_labels = parse_dataset(en_trf_path)
    zh_train_images, zh_train_labels, zh_test_images, zh_test_labels = parse_dataset(zh_trf_path)
    print(f'TFRecords loaded.')

    en_identifier = LPIdentification(en_trf_path, en_train_images, en_train_labels, en_test_images, en_test_labels)
    zh_identifier = LPIdentification(zh_trf_path, zh_train_images, zh_train_labels, zh_test_images, zh_test_labels)
    print(f'Identifier ready.')

    train_new = False
    en_identifier.load_h5_model(train_new)
    zh_identifier.load_h5_model(train_new)
    if train_new:
        print(f'Successfully trained new .h5 model.')
    else:
        print(f'Successfully loaded existing .h5 model.')


    files = os.listdir(LP_TEST_IMGS_PATH)
    file_cnt = len(files)
    counter = 1
    cols = 8
    rows = math.floor(file_cnt / cols)
    size = (rows, cols)

    fig = plt.figure(figsize=(25.6, 14.4), dpi=100)

    for i in files[:-3]:
        current_file = LP_TEST_IMGS_PATH + i
        print(f'Loaded {current_file.split("/")[-1]}.')


    lpltr = LPLocator(current_file)
    img_lp_highlighted, img_lp_cropped, char_imgs = lpltr.rough_process()
    img, shadow_image, sliced_photos = lpltr.return_image()


    if len(char_imgs) == 0:
        print(f'Failed to identify the license...')
    else:
        result = zh_identifier.identify_chars([char_imgs[0]])
        result.append(en_identifier.identify_chars(char_imgs[1:]))

        # plt.imshow(img_lp_highlighted)
        #plt.imshow(shadow_image)
       # plt.imshow(sliced_photos)
       # plt.title(result_coversion(result))
        #plt.show()

    return img_lp_highlighted, shadow_image, sliced_photos, result_coversion(result)
        # os.system("pause")


    # os.system("pause")

    # img = CLPDImage(current_file)
    #
    # img.get_csv_data(csv_data[img.get_id()])
    #
    # img.preprocess()
    # # img.write_all()
