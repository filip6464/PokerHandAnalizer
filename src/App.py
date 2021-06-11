import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.DataParser import DataParser, normalize_x_data, normalize_y_data
from src.DataPlotter import plot_data, plot_missing_data
from src.Pca import pcaFunction

data_parser = DataParser()
x_testing_data, y_testing_data, x_training_data, y_training_data, subsets = data_parser.import_data()

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

print("------------------")
print("Machine Learning")
# machine learning algorithms
le = preprocessing.LabelEncoder()

decision_tree_confusion_matrix = []
decision_tree_reports = []

k_neighbors_confusion_matrix = []
k_neighbors_reports = []

MLP_confusion_matrix = []
MLP_reports = []

print("Learning is proceed... Please wait a while")
for i in range(8):
    print("Step ", i+1, "/8")
    x_train = normalize_x_data(subsets[i][:, :-1])
    y_train = normalize_y_data(subsets[i][:, -1])

    x_test = normalize_x_data(subsets[(i+1) % 8][:, :-1])
    y_test = normalize_y_data(subsets[(i+1) % 8][:, -1])
    for x in range(6):
        x_test = np.concatenate((x_test, normalize_x_data(subsets[(i+2+x) % 8][:, :-1])))
        y_test = np.concatenate((y_test, normalize_y_data(subsets[(i+2+x) % 8][:, -1])))

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    #decision tree
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    decision_tree = decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)
    matrix = confusion_matrix(y_pred, y_test)
    report = classification_report(y_pred, y_test, zero_division=0)

    decision_tree_confusion_matrix.append(matrix)
    decision_tree_reports.append(report)

    #KNeighbors
    K_classifier = KNeighborsClassifier(n_neighbors=2)
    K_classifier.fit(x_train, y_train)
    y_pred = K_classifier.predict(x_test)
    matrix = confusion_matrix(y_pred, y_test)
    report = classification_report(y_pred, y_test, zero_division=0)

    k_neighbors_confusion_matrix.append(matrix)
    k_neighbors_reports.append(report)

    MLP_classifier = MLPClassifier(alpha=1e-2, max_iter=250, random_state=1)
    MLP_classifier.fit(x_train, y_train)
    y_pred = MLP_classifier.predict(x_test)
    matrix = confusion_matrix(y_pred, y_test)
    report = classification_report(y_pred, y_test, zero_division=0)

    MLP_confusion_matrix.append(matrix)
    MLP_reports.append(report)


print("------------------------")

print("Decision tree - result index 0 \n")
print("Confusion_matrix \n")
print(decision_tree_confusion_matrix[0])
print("Report \n")
print(decision_tree_reports[0])

print("K Neighbors - result index 0 \n")
print("Confusion_matrix \n")
print(k_neighbors_confusion_matrix[0])
print("Report \n")
print(k_neighbors_reports[0])

print("MLP - result index 0 \n")
print("Confusion_matrix \n")
print(MLP_confusion_matrix[0])
print("Report \n")
print(MLP_reports[0])



