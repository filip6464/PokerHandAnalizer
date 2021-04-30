import numpy as np


def normalize_x_data(data_array):
    even_min = 1
    even_max = 13
    odd_min = 1
    odd_max = 4

    # even columns
    data_array[:, 1::2] = (data_array[:, 1::2] - even_min) / (even_max - even_min)

    #odd columns
    data_array[:, ::2] = (data_array[:, ::2] - odd_min) / (odd_max - odd_min)

    return data_array


def normalize_y_data(data_array):
    min_value = 0
    max_value = 9

    data_array[:] = (data_array[:] - min_value) / (max_value - min_value)

    return data_array

class DataParser:
    testing_file_path = r'../Data/poker-hand-testing.data'
    training_file_path = r'../Data/poker-hand-training-true.data'

    def import_data(self):
        # Loading testing data form file
        testing_data = np.loadtxt(fname=self.testing_file_path, delimiter=',')
        testing_data = np.unique(testing_data, axis=0)  # making all rows unique

        x_testing_data = testing_data[:, :-1]
        y_testing_data = testing_data[:, -1]

        # Loading training data form file
        training_data = np.loadtxt(fname=self.training_file_path, delimiter=',')
        training_data = np.unique(training_data, axis=0)  # making all rows unique
        x_training_data = training_data[:, :-1]
        y_training_data = training_data[:, -1]

        return x_testing_data, y_testing_data, x_training_data, y_training_data
