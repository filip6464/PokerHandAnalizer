from sklearn.ensemble import RandomForestClassifier
from src.DataParser import DataParser

data_parser = DataParser()
x_testing_data, y_testing_data, x_training_data, y_training_data = data_parser.import_data()

print("x_testing_data")
print(x_testing_data)

print("y_testing_data")
print(y_testing_data)

print("x_training_data")
print(x_training_data)

print("y_training_data")
print(y_training_data)
