import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Naive Bayes Classifier
# get training data from csv
data = np.loadtxt("classification_dataset_training.csv", dtype=int, skiprows=1, delimiter=',',)

indexRating = len(data[0])-1


X_data = data[:,range(1,indexRating)]
y_data = data[:,indexRating]

gnb = GaussianNB()
y_pred = gnb.fit(X_data, y_data).predict(X_data)


# Log-Bernoulli loss or prediction accurary
print("Number of mislabeled points out of a total %d points : %d"  % (X_data.shape[0],(y_data != y_pred).sum()))