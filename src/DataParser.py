import numpy as np


class DataParser:
    testing_file_path = r'../Data/poker-hand-testing.data'
    training_file_path = r'../Data/poker-hand-training-true.data'

    def import_data(self):
        # Loading testing data form file
        testing_data = np.loadtxt(fname=self.testing_file_path, delimiter=',')
        x_testing_data = testing_data[:, :-1]
        y_testing_data = testing_data[:, -1]

        # Loading training data form file
        training_data = np.loadtxt(fname=self.training_file_path, delimiter=',')
        x_training_data = training_data[:, :-1]
        y_training_data = training_data[:, -1]

        return x_testing_data, y_testing_data, x_training_data, y_training_data
