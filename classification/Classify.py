from collections import UserString
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from classification.GaussianNaiveBayesClassifier import GaussianNaiveBayesClassifier
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import MultinomialNB

cross_valid_cache = []
def report_performance(valid_total, valid_incorrect, classifiers):
    for i in range(0, len(classifiers)):
        print(classifiers[i]['name'])
        print("\tvalidation accuracy= %d/%d incorrect = %f percent" %
              (valid_incorrect[i], valid_total[i], valid_incorrect[i] / valid_total[i]))


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


def plot(classifiers_name, accuracy_result, title):
    x = np.linspace(0, len(classifiers_name), len(classifiers_name))
    plt.bar(x, accuracy_result)
    x = np.add(x, 0.35)
    plt.xticks(x, classifiers_name)
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy error")
    axes = plt.gca()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def setup_classifiers():
    lr = {'classifier': LogisticRegression(n_jobs=4), 'name': "LogisticRegression"}
    gnb = {'classifier': GaussianNB(), 'name': "GaussianNB"}
    clf = {'classifier': MultinomialNB(), 'name': "MultinomialNB"}
    my_gnb = {'classifier': GaussianNaiveBayesClassifier(), 'name': "GaussianNaiveBayesClassifier"}
    knn200 = {'classifier': KNeighborsClassifier(n_neighbors=200, n_jobs=4), 'name': "200-NeighborsClassifier"}
    knn50 = {'classifier': KNeighborsClassifier(n_neighbors=50, n_jobs=4), 'name': "50-NeighborsClassifier"}
    knn1 = {'classifier': KNeighborsClassifier(n_neighbors=1, n_jobs=4), 'name': "1-NeighborsClassifier"}
    knn5 = {'classifier': KNeighborsClassifier(n_neighbors=5, n_jobs=4), 'name': "5-NeighborsClassifier"}
    knn10 = {'classifier': KNeighborsClassifier(n_neighbors=10, n_jobs=4), 'name': "10-NeighborsClassifier"}
    knn100 = {'classifier': KNeighborsClassifier(n_neighbors=100, n_jobs=4), 'name': "100-NeighborsClassifier"}
    # classifiers = [gnb, my_gnb, lr, knn50, knn20, clf]
    classifiers = [knn1, knn5, knn10, knn50, knn200, knn100]
    classifiers_name = []
    for classifier in classifiers:
        classifiers_name.append(classifier["name"])
    return classifiers, classifiers_name


def perform_cross_validation(training_features, training_label, classifiers, nb_iter, cross_valid_cache ):
    list_total, list_incorrect = [0] * len(classifiers), [0] * len(classifiers)

    # Random repartition for training & validation training_data in order to perform cross validation
    if len(cross_valid_cache) == 0:
        cross_valid_cache = ShuffleSplit(len(training_features[:, 0]), n_iter=nb_iter, test_size=0.20)
    for train_indexes, valid_indexes in cross_valid_cache:
        # Split in validation & training set
        valid_features, train_features = training_features[valid_indexes, :], training_features[train_indexes, :]
        valid_label, train_label = training_label[valid_indexes], training_label[train_indexes]

        # Classify for each randomly picked training and validation set
        total, incorrect = classify_all(classifiers,
                                        train_features, train_label,
                                        valid_features, valid_label)

        list_total = np.add(list_total, total)
        list_incorrect = np.add(list_incorrect, incorrect)
    return np.divide(list_incorrect, list_total), cross_valid_cache


def plot_all(results, classifiers, title):
    for index, classifier in enumerate(classifiers):
        plt.plot(range(1, 51), results[:, index], label=classifier["name"])
    plt.xlabel("Features number")
    plt.ylabel("Accuracy error")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


def main():
    # Read  data
    training_features = np.loadtxt("classification_dataset_training.csv",
                                   dtype=int, skiprows=1, delimiter=',', usecols=range(1, 51), )
    training_label = np.loadtxt("classification_dataset_training.csv",
                                dtype=int, skiprows=1, delimiter=',', usecols=range(51, 52), )
    test_features = np.loadtxt("classification_dataset_testing.csv",
                               dtype=int, skiprows=1, delimiter=',', usecols=range(1, 51), )
    test_label = np.loadtxt("classification_dataset_testing_solution.csv",
                            dtype=int, skiprows=1, delimiter=',', usecols=range(1, 2), )
    features_names = np.loadtxt("classification_dataset_training.csv",
                                dtype=UserString, delimiter=',', usecols=range(1, 51), )[0, :]

    # Setup Classifiers
    classifiers, classifiers_name = setup_classifiers()

    test_results = np.zeros((len(training_features[0, :]), len(classifiers)))
    valid_results = np.zeros((len(training_features[0, :]), len(classifiers)))
    choosen_features = [[]] * len(training_features[0, :])
    cross_valid_cache = []
    # Perform cross validation for nb features with nb in [1, 50]

    for index, nb_features in enumerate(range(1, len(training_features[0, :]) + 1)):
        print(nb_features)
        # Feature extraction
        mask_array = feature_selection(training_features, training_label, nb_features)

        # Keep best features
        training_features_tmp = training_features[:, mask_array]
        test_features_tmp = test_features[:, mask_array]
        choosen_features[nb_features - 1] = features_names[mask_array]

        nb_iter = 100

        accuracy_result, cross_valid_cache = perform_cross_validation(training_features_tmp, training_label, classifiers, nb_iter, cross_valid_cache)
        valid_results[index] = accuracy_result

        # Classify for each randomly picked training and validation set
        total, incorrect = classify_all(classifiers,
                                        training_features_tmp, training_label,
                                        test_features_tmp, test_label)
        test_results[index] = np.divide(incorrect, total)

    plot_all(results=valid_results, classifiers=classifiers, title="Average accuracy error on cross validation process")
    plot_all(results=test_results, classifiers=classifiers, title="Accuracy error on test set")
    '''
    with open("features-latex.txt", "tw", encoding='utf-8') as text_file:
        for i in range(0, 50):
            #{print("For", i + 1, " features, best one are: ")
            #print(choosen_features[i])
            # Latex format !
            j = i+1
            text_file.write("%d & " % j)
            for index, feature in enumerate(choosen_features[i]):
                if index == len(choosen_features[i]) - 1:
                    text_file.write("%s " % feature)
                else:
                    text_file.write("%s, " % feature)
            print("\\\ \hline \n", file=text_file)
    '''


if __name__ == "__main__":
    main()

# print("Cross Validation result:")
'''
# report_performance(list_total, list_incorrect, classifiers)
# title = "Comparison between classifiers for validation accuracy error and %d features" % len(features_names)
# plot(classifiers_name, accuracy_result, title)
'''
# Perform best classifier on test training_data
'''
index_best_classifier = accuracy_result.argmin()
print("Best classifier is ", classifiers[index_best_classifier]['name'])
print("\n\nActual performance on test set:")
'''
# report_performance(total, incorrect, classifiers)
# title = "Comparison between classifiers for test accuracy error and %d features" % len(features_names)
# plot(classifiers_name, accuracy_result, title)
