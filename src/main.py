import PyQt5

from src.LPLocator.LPLocator import LPLocator
from CLPD.CLPDImage import CLPDImage
from CSVReader.CSVReader import CSVReader
from Dataset.DatasetManip import generate_dataset, parse_dataset
from LPIdentification.LPIdentification import LPIdentification

BASE_DATASET_PATH = '../CLPD_1200/'
TFR_PATH = '../model/license_plate_test.tfrecords'

if __name__ == "__main__":

    print("|--License Plate Recognition--|")

    generate_dataset(TFR_PATH)
    train_images, train_labels = parse_dataset(TFR_PATH)
    identifier = LPIdentification(TFR_PATH, train_images, train_labels)
    identifier.execute_identification()

    # desired_csv_size = 100
    # cr = CSVReader(desired_csv_size)
    #
    # col_titles = cr.get_csv_title()
    # csv_data = cr.get_csv_data()
    #
    # imgs = []
    # batch = 2
    # batch_size = 10
    # for i in range((batch - 1) * batch_size, batch * batch_size):
    #
    #     current_file = BASE_DATASET_PATH + str(i) + '.jpg'
    #
    #     img = CLPDImage(current_file)
    #
    #     img.get_csv_data(csv_data[img.get_id()])
    #
    #     img.preprocess()
        # img.write_all()

        # lpltr = LPLocator(current_file)
        # lpltr.process()

    # seq = 1
    # size = 9
    # images = []
    # for i in range(seq, seq + size - 1):
    #     # img = LPImage('../img/' + str(i) + '.jpg')
    #     img = LPImagePlus('../img/' + str(i) + '.jpg')
    #     img.process()
    #     images.append(img)
    # # img = LPImage('../img/12.jpg')
    # # img.process()

