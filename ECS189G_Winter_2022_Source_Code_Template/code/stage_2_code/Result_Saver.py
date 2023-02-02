'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
import pickle
import os

class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None
    counter = 1
    
    def save(self):
        print('saving results...')

        while os.path.exists(path):
            path = self.result_destination_file_name + " (" + str(self.counter) + ")"
            self.counter += 1

        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'wb')
        pickle.dump(self.data, f)
        f.close()
