'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        print('training set size:', len(data['train']), 'testing set size:',
              len(data['test']))
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for pair in data['train']:
            X_train.append(pair['image'])
            y_train.append(pair['label'])

        for pair in data['test']:
            X_test.append(pair['image'])
            y_test.append(pair['label'])

        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

        # x_train is the value of the top-level dictionary for key 'train',
        # then the values for all of the keys 'image' in all of the dictionaries