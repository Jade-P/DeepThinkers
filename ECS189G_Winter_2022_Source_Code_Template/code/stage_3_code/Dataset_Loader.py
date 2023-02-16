'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        train_data = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)
        X_train = train_data.iloc[:, 1:]
        y_train = train_data.iloc[:, 0]

        test_data = pd.read_csv(self.dataset_source_folder_path + 'test.csv')
        X_test = test_data.iloc[:, 1:]
        y_test = test_data.iloc[:, 0]

        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}