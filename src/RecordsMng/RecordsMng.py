from src.Dataset.DatasetManip import generate_records, parse_dataset
from src.LPIdentification.LPIdentification import LPIdentification

EN_TFR_PATH = '../model/license_plate_en.tfrecords'
ZH_TFR_PATH = '../model/license_plate_zh.tfrecords'


class RecordsMng(object):
    def __init__(self, gen_new, train_new):

        if gen_new:
            generate_records('en')
            generate_records('zh')
            print(f'TFRecords generated.')

        en_train_images, en_train_labels, en_test_images, en_test_labels = parse_dataset(EN_TFR_PATH)
        zh_train_images, zh_train_labels, zh_test_images, zh_test_labels = parse_dataset(ZH_TFR_PATH)
        print(f'TFRecords loaded.')

        self.en_identifier = LPIdentification(EN_TFR_PATH, en_train_images, en_train_labels, en_test_images,
                                              en_test_labels)
        self.zh_identifier = LPIdentification(ZH_TFR_PATH, zh_train_images, zh_train_labels, zh_test_images,
                                              zh_test_labels)
        print(f'Identifiers ready.')

        self.en_identifier.load_h5_model(train_new)
        self.zh_identifier.load_h5_model(train_new)
        if train_new:
            print(f'Successfully trained new .h5 model.')
        else:
            print(f'Successfully loaded existing .h5 model.')

    def get_identifiers(self):
        return self.en_identifier, self.zh_identifier
