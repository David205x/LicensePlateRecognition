import os

import PyQt5

from src.LPLocator.LPLocator import LPLocator
from CSVReader.CSVReader import CSVReader
from Dataset.DatasetManip import generate_dataset, parse_dataset
from LPIdentification.LPIdentification import LPIdentification

BASE_DATASET_PATH = '../CLPD_1200/'
LP_TEST_IMGS_PATH = '../lp/'

if __name__ == "__main__":

    print("|--License Plate Recognition--|")

    # desired_csv_size = 100
    # cr = CSVReader(desired_csv_size)
    #
    # col_titles = cr.get_csv_title()
    # csv_data = cr.get_csv_data()
    # print(f'CSV data loaded.')

    en_trf_path = generate_dataset('en')
    zh_trf_path = generate_dataset('zh')

    en_train_images, en_train_labels, en_test_images, en_test_labels = parse_dataset(en_trf_path)
    zh_train_images, zh_train_labels, zh_test_images, zh_test_labels = parse_dataset(zh_trf_path)
    print(f'TFRecords loaded.')

    en_identifier = LPIdentification(en_trf_path, en_train_images, en_train_labels, en_test_images, en_test_labels)
    zh_identifier = LPIdentification(zh_trf_path, zh_train_images, zh_train_labels, zh_test_images, zh_test_labels)
    print(f'Model training ready.')

    train_new = False
    en_identifier.load_h5_model(train_new)
    zh_identifier.load_h5_model(train_new)
    print(f'Successfully loaded .h5 model.')

    files = os.listdir(LP_TEST_IMGS_PATH)

    for i in files[:-3]:
        current_file = LP_TEST_IMGS_PATH + i
        print(f'Loaded {current_file}.')

        lpltr = LPLocator(current_file)
        img_lp_highlighted, img_lp_cropped, char_imgs = lpltr.rough_process()

        if len(char_imgs) == 0:
            print(f'Failed to identify the license...')
        else:
            result = zh_identifier.identify_chars([char_imgs[0]])
            result.append(en_identifier.identify_chars(char_imgs[1:]))
            print(result)

        # os.system("pause")

        # img = CLPDImage(current_file)
        #
        # img.get_csv_data(csv_data[img.get_id()])
        #
        # img.preprocess()
        # # img.write_all()

