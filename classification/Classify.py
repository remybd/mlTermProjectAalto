import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from classification.GaussianNaiveBayesClassifier import GaussianNaiveBayesClassifier
import scipy as sp


def report_performance(valid_total, valid_incorrect, classifiers):
    for i in range(0, len(classifiers)):
        print(classifiers[i]['name'])
        print("\tvalidation accuracy= %d/%d incorrect = %f percent" %
              (valid_incorrect[i], valid_total[i], valid_incorrect[i]/valid_total[i]))


def log_bernoulli_loss(y, y_pred, p=0.9):
    sum = 0
    for i in range(len(y)):
        y_pred[i] = np.maximum(p, y_pred[i])
        y_pred[i] = np.minimum(1 - p, y_pred[i])
        p = y_pred[i]
        # r = 1 if y[i] == y_pred[i] else 0
        r = y[i]
        sum = sum + r * np.log(p) + np.subtract(1, r) * np.log(np.subtract(1, p))
    result = sum * -1/len(y)
    print("Log Bernoulli loss error = ", result)
    return result


def accuracy_ratio(X, y, y_pred):
    total, correct = (X.shape[0], (y != y_pred).sum())
    return total, correct


def classify(classifier, train_features, train_label, valid_features, valid_label):
    label_pred = classifier.fit(train_features, train_label).predict(valid_features)
    total, correct = accuracy_ratio(valid_features, valid_label, label_pred)
    return total, correct


def split_features_label(data):
    width = len(data[0, :]) -1
    return data[:, range(0, width)], data[:, width]


def classify_all(classifiers, train_data_features, train_data_label, valid_data_features, valid_data_label):
    list_total = [0] * len(classifiers)
    list_incorrect = [0] * len(classifiers)
    # Compute accuracy error for each classifier
    for index, item in enumerate(classifiers):
        total, incorrect = classify(classifiers[index]['classifier'],
                                    train_data_features,
                                    train_data_label,
                                    valid_data_features,
                                    valid_data_label)
        list_total[index] += total
        list_incorrect[index] += incorrect
    return list_total, list_incorrect


def main():
    # Read  data
    training_data = np.loadtxt("classification_dataset_training.csv",
                               dtype=int, skiprows=1, delimiter=',', usecols=range(1, 52),)
    test_data_features = np.loadtxt("classification_dataset_testing.csv",
                                    dtype=int, skiprows=1, delimiter=',', usecols=range(1, 51),)
    test_data_label = np.loadtxt("classification_dataset_testing_solution.csv",
                                 dtype=int, skiprows=1, delimiter=',', usecols=range(1, 2),)

    # Setup Classifiers
    lr = {'classifier': LogisticRegression(), 'name': "LogisticRegression"}
    gnb = {'classifier': GaussianNB(), 'name': "GaussianNB"}
    svc = {'classifier': LinearSVC(C=1.0), 'name': "LinearSVC"}
    mygnb = {'classifier': GaussianNaiveBayesClassifier(), 'name': "GaussianNaiveBayesClassifier (from scratch)"}
    rfc = {'classifier': RandomForestClassifier(n_estimators=100), 'name': "RandomForestClassifier"}
    classifiers = [mygnb, gnb, svc, rfc, lr]

    nb_iter = 10

    # Random repartition for training & validation training_data in order to perform cross validation
    rs = ShuffleSplit(len(training_data[:, 0]), n_iter=nb_iter, test_size=0.20)
    list_total = [0] * len(classifiers)
    list_incorrect = [0] * len(classifiers)
    for train_indexes, valid_indexes in rs:
        # Split in validation & training set
        valid_data, train_data = training_data[valid_indexes, :], training_data[train_indexes, :]

        train_data_features, train_data_label = split_features_label(train_data)
        valid_data_features, valid_data_label = split_features_label(valid_data)
        # Classify for each randomly picked training and validation set
        total, incorrect = classify_all(classifiers,
                                        train_data_features, train_data_label,
                                        valid_data_features, valid_data_label)

        list_total = np.add(list_total, total)
        list_incorrect = np.add(list_incorrect, incorrect)

    print("Cross Validation result:")
    report_performance(list_total, list_incorrect, classifiers)

    '''
    # Perform best classifier on test training_data
    '''
    accuracy_result = np.divide(list_incorrect, list_total)
    index_best_classifier = accuracy_result.argmin()

    print("Best classifier is ", classifiers[index_best_classifier]['name'])
    print("\n\nActual performance on test set:")

    train_data_features, train_data_label = split_features_label(training_data)

    # Classify for each randomly picked training and validation set
    total, incorrect = classify_all(classifiers,
                                    train_data_features, train_data_label,
                                    test_data_features, test_data_label)
    report_performance(total, incorrect, classifiers)

if __name__ == "__main__":
    main()