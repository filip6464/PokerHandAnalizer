import numpy as np


class DataParser:
    testing_file_path = r'../Data/poker-hand-testing.data'
    training_file_path = r'../Data/poker-hand-training-true.data'

    def import_data(self):
        # Loading testing data form file
        print("Testing data")
        testing_data = np.loadtxt(fname=self.testing_file_path, delimiter=',')
        print(testing_data)

        # Loading training data form file
        print("Training data")
        training_data = np.loadtxt(fname=self.training_file_path, delimiter=',')
        print(training_data)

        return testing_data, training_data
