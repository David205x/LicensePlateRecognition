import PyQt5

from src.LPLocator.LPLocator import LPLocator
from CLPD.CLPDImage import CLPDImage
from CSVReader.CSVReader import CSVReader

BASE_DATASET_PATH = '../CLPD_1200/'

if __name__ == "__main__":

    print("--License Plate Recognition--")

    desired_csv_size = 300
    cr = CSVReader(desired_csv_size)

    col_titles = cr.get_csv_title()
    csv_data = cr.get_csv_data()

    imgs = []

    for i in range(11, 16):

        current_file = BASE_DATASET_PATH + str(i) + '.jpg'
        lpltr = LPLocator(current_file)
        # img = CLPDImage(current_file)
        #
        # img.get_csv_data(csv_data[img.get_id()])
        #
        # img.preprocess()
        # img.write_all()

        lpltr.process()

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

