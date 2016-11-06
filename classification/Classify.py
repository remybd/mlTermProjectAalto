import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def predict(classifier, X, y):
    y_pred = classifier.fit(X, y).predict(X)
    print(classifier)
    # Log-Bernoulli loss or prediction accurary
    print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0], (y != y_pred).sum()))

# Read data
data = np.loadtxt("classification_dataset_training.csv", dtype=int, skiprows=1, delimiter=',',)
indexRating = len(data[0])-1
X_data = data[:, range(1, indexRating)]
y_data = data[:, indexRating]



lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)

predict(lr,X_data,y_data)
predict(gnb,X_data,y_data)
predict(svc,X_data,y_data)
predict(rfc,X_data,y_data)




