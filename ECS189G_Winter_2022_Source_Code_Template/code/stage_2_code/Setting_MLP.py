'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_MLP(setting):
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train = np.array(loaded_data['X_train'])
        X_test = np.array(loaded_data['X_test'])
        y_train = np.array(loaded_data['y_train'])
        y_test = np.array(loaded_data['y_test'])

        train_accuracy = []
        test_accuracy = []

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test},
                            'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
        learned_result = self.method.run()
        self.method.save_plot()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

