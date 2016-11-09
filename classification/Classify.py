from collections import UserString

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from classification.GaussianNaiveBayesClassifier import GaussianNaiveBayesClassifier
from sklearn.feature_selection import RFE
import scipy as sp


def report_performance(valid_total, valid_incorrect, classifiers):
    for i in range(0, len(classifiers)):
        print(classifiers[i]['name'])
        print("\tvalidation accuracy= %d/%d incorrect = %f percent" %
              (valid_incorrect[i], valid_total[i], valid_incorrect[i]/valid_total[i]))


def accuracy_ratio(X, y, y_pred):
    total, correct = (X.shape[0], (y != y_pred).sum())
    return total, correct


def classify(classifier, train_features, train_label, valid_features, valid_label):
    label_pred = classifier.fit(train_features, train_label).predict(valid_features)
    total, correct = accuracy_ratio(valid_features, valid_label, label_pred)
    return total, correct


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


def feature_selection(x, y, nb_features):
    model = LogisticRegression()
    rfe = RFE(model, nb_features)
    fit = rfe.fit(x, y)
    return fit.support_


def main():
    # Read  data
    training_features = np.loadtxt("classification_dataset_training.csv",
                               dtype=int, skiprows=1, delimiter=',', usecols=range(1, 51),)
    training_label = np.loadtxt("classification_dataset_training.csv",
                                   dtype=int, skiprows=1, delimiter=',', usecols=range(51, 52), )
    test_features = np.loadtxt("classification_dataset_testing.csv",
                                    dtype=int, skiprows=1, delimiter=',', usecols=range(1, 51),)
    test_label = np.loadtxt("classification_dataset_testing_solution.csv",
                                 dtype=int, skiprows=1, delimiter=',', usecols=range(1, 2),)
    features_names = np.loadtxt("classification_dataset_training.csv",
                                 dtype=UserString, delimiter=',', usecols=range(1, 51),)[0, :]

    # Feature extraction
    nb_features = 40
    mask_array = feature_selection(training_features, training_label, nb_features)

    # Keep best features
    training_features = training_features[:, mask_array]
    choosen_features = features_names[mask_array]
    test_features = test_features[:, mask_array]

    # Setup Classifiers
    lr = {'classifier': LogisticRegression(), 'name': "LogisticRegression"}
    gnb = {'classifier': GaussianNB(), 'name': "GaussianNB"}
    svc = {'classifier': LinearSVC(C=1.0), 'name': "LinearSVC"}
    my_gnb = {'classifier': GaussianNaiveBayesClassifier(), 'name': "GaussianNaiveBayesClassifier (from scratch)"}
    rfc = {'classifier': RandomForestClassifier(n_estimators=100), 'name': "RandomForestClassifier"}
    classifiers = [my_gnb, gnb, svc, rfc, lr]

    nb_iter = 10

    list_total = [0] * len(classifiers)
    list_incorrect = [0] * len(classifiers)
    # Random repartition for training & validation training_data in order to perform cross validation
    rs = ShuffleSplit(len(training_features[:, 0]), n_iter=nb_iter, test_size=0.20)
    for train_indexes, valid_indexes in rs:
        # Split in validation & training set
        valid_features, train_features = training_features[valid_indexes, :], training_features[train_indexes, :]
        valid_label, train_label = training_label[valid_indexes], training_label[train_indexes]

        # Classify for each randomly picked training and validation set
        total, incorrect = classify_all(classifiers,
                                        train_features, train_label,
                                        valid_features, valid_label)

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

    # Classify for each randomly picked training and validation set
    total, incorrect = classify_all(classifiers,
                                    training_features, training_label,
                                    test_features, test_label)
    report_performance(total, incorrect, classifiers)

if __name__ == "__main__":
    main()