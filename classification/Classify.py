import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
import scipy as sp

def logloss(act, pred):
    epsilon = 0.9
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


def report_performance(valid_error,valid_total, valid_incorrect, classifiers):
    for i in range(0, len(classifiers)):
        print(classifiers[i]['name'])
        print("\tvalidation logloss=",valid_error[i])
        print("\tvalidation accuracy= %d/%d incorrect = %f percent" % (valid_incorrect[i], valid_total[i], valid_incorrect[i]/valid_total[i]))


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
    # print("Number of mislabeled points out of a total %d points : %d" % (total, correct))
    return total, correct


def get_errors(classifier, X, y, X_test, y_test):
    y_pred = classifier.fit(X, y).predict(X_test)
    # print(classifier)
    # Log-Bernoulli loss or prediction accurary
    #
    # log_bernoulli_loss(y,y_pred)
    total, correct = accuracy_ratio(X_test, y_test, y_pred)
    return logloss(y_test, y_pred), total, correct


def get_x_y(data, index_rating):
    return data[:, range(1, index_rating)], data[:, index_rating]

# Read data
data = np.loadtxt("classification_dataset_training.csv", dtype=int, skiprows=1, delimiter=',',)
indexRating = len(data[0])-1
X_data = data[:, range(1, indexRating)]
y_data = data[:, indexRating]

# Setup Classifiers
lr = {'classifier': LogisticRegression(), 'name': "LogisticRegression"}
gnb = {'classifier': GaussianNB(), 'name': "GaussianNB"}
svc = {'classifier': LinearSVC(C=1.0), 'name': "LinearSVC"}
# mygnb = {'classifier': MyNaiveBayesClassifier(), 'name': "MyNaiveBayesClassifier"}
rfc = {'classifier': RandomForestClassifier(n_estimators=100), 'name': "RandomForestClassifier"}
classifiers = [gnb, svc, rfc, lr]


nb_iter = 10
valid_log_loss = [0] * len(classifiers)
valid_incorrect = [0] * len(classifiers)
valid_total = [0] * len(classifiers)
# Random repartition for training & validation data in order to perform cross validation
rs = ShuffleSplit(len(data[:, 0]), n_iter=nb_iter, test_size=0.25)
for train_indexes, test_indexes in rs:
    # Split in validation & training set
    valid_data, train_data = data[test_indexes, :], data[train_indexes, :]
    X_train_data, y_train_data = get_x_y(train_data, indexRating)
    X_valid_data, y_valid_data = get_x_y(valid_data, indexRating)
    # Compute log loss error for each classifier
    for index, item in enumerate(classifiers):

        log_loss, total, incorrect = get_errors(classifiers[index]['classifier'], X_train_data, y_train_data, X_valid_data, y_valid_data)
        valid_log_loss[index] += log_loss
        valid_incorrect[index] += incorrect
        valid_total[index] += total


valid_log_loss = np.divide(valid_log_loss, nb_iter)
report_performance(valid_log_loss, valid_total, valid_incorrect, classifiers)

'''
# Perform best classifier on test data
'''