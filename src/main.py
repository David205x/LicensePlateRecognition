import PyQt5

from src.LPLocator.LPLocator import LPLocator
from CSVReader.CSVReader import CSVReader
from Dataset.DatasetManip import generate_dataset, parse_dataset
from LPIdentification.LPIdentification import LPIdentification

BASE_DATASET_PATH = '../CLPD_1200/'
TFR_PATH = '../model/license_plate_test.tfrecords'
LP_TEST_IMGS_PATH = '../lp/'

if __name__ == "__main__":

    print("|--License Plate Recognition--|")

    # desired_csv_size = 100
    # cr = CSVReader(desired_csv_size)
    #
    # col_titles = cr.get_csv_title()
    # csv_data = cr.get_csv_data()
    # print(f'CSV data loaded.')

    generate_dataset(TFR_PATH)
    train_images, train_labels, test_images, test_labels = parse_dataset(TFR_PATH)
    print(f'TFRecords loaded.')
    identifier = LPIdentification(TFR_PATH, train_images, train_labels, test_images, test_labels)
    print(f'Model training ready.')
    identifier.load_h5_model(False)
    print(f'Successfully loaded .h5 model.')

    # batch = 2
    # batch_size = 10
    # for i in range((batch - 1) * batch_size, batch * batch_size):
    for i in range(11, 13):
        current_file = LP_TEST_IMGS_PATH + str(i) + '.jpg'
        print(f'Loaded {current_file}.')

        lpltr = LPLocator(current_file)
        img_lp_highlighted, img_lp_cropped, char_imgs = lpltr.rough_process()

        if char_imgs is None:
            print(f'Failed to identify the license...')
        else:
            result = identifier.identify_chars(char_imgs)
            print(result)

        # img = CLPDImage(current_file)
        #
        # img.get_csv_data(csv_data[img.get_id()])
        #
        # img.preprocess()
        # # img.write_all()

