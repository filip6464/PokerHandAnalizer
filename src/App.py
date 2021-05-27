from sklearn.ensemble import RandomForestClassifier
from src.DataParser import DataParser, normalize_x_data, normalize_y_data
from src.DataPlotter import plot_data, plot_missing_data
from src.Pca import pcaFunction

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

plot_data(y_testing_data, 'Distribution of card layouts in the training data set', '../plots/Y_testing_data_plot.png')
plot_data(y_training_data, 'Distribution of card layouts in the test data set', '../plots/Y_training_data_plot.png')

plot_missing_data(x_training_data, x_testing_data, 'Amount of poker hand combinations', '../plots/combinations_plot.png')

# normalize data
x_normalize_testing_data = normalize_x_data(x_testing_data)
x_normalize_training_data = normalize_x_data(x_training_data)
y_normalize_testing_data = normalize_y_data(y_testing_data)
y_normalize_training_data = normalize_y_data(y_training_data)


print("x_normalize_testing_data")
print(x_normalize_testing_data)

print("x_normalize_training_data")
print(x_normalize_testing_data)

print("y_normalize_testing_data")
print(y_normalize_testing_data)

print("y_normalize_training_data")
print(y_normalize_training_data)

pcaFunction(x_normalize_testing_data, '../plots/scree_plot.png', '../plots/pca_plot.png')
