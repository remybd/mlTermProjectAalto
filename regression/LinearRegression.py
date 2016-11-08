import numpy as np;

class LinearRegression:

    def __init__(self):
        self.weight = []

    def fit(self,X, Y):
        # X[(X'X)^(-1) X' Y]
        self.weight = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y));
        return self.weight

    def predict(self,X):
        return np.dot(X,self.weight)

