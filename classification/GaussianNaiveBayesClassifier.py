import numpy as np


class GaussianNaiveBayesClassifier:
    def __init__(self):
        self._fittedData = None

    # Split rows according to the value in y
    # Will fail if one class is not present
    def _split_by_class(self, features, label):
        splitted = {0: [], 1: []}
        for i in range(0, len(label)):
            if label[i] == 0:
                splitted[0].append(features[i])
            else:
                splitted[1].append(features[i])
        return splitted

    def _actual_fit(self, features):
        # for each attribute we compute both the mean and the variance
        summaries = [(np.mean(attribute), np.var(attribute)) for attribute in zip(*features)]
        del summaries[-1]
        return summaries

    def fit(self, features, label):
        splitted = self._split_by_class(features, label)
        self._fittedData = {}
        for classLabel in splitted:
            self._fittedData[classLabel] = self._actual_fit(splitted[classLabel])
        return self

    def _compute_proba(self, row_values, mean, variance):
        exponent = np.exp(-(np.power(row_values - mean, 2) / (2 * np.power(variance, 2))))
        return (1 / (np.sqrt(2 * np.pi) * variance)) * exponent

    # Compute the probability of each class
    def _compute_class_proba(self, features):
        probabilities = {}
        for classLabel in self._fittedData:
            probabilities[classLabel] = 1
            for index, meanVariance in enumerate(self._fittedData[classLabel]):
                row_values = features[index]
                probabilities[classLabel] *= self._compute_proba(row_values,
                                                                 mean=meanVariance[0],
                                                                 variance=meanVariance[1])
        return probabilities

    # Will fail if fit has not be called before
    def predict(self, X):
        result = []
        # for each sample we predict the class
        for i in range(len(X)):
            probabilities = self._compute_class_proba(X[i])
            best_class_label, best_proba = None, -1
            for classLabel in probabilities:
                if best_class_label is None or probabilities[classLabel] > best_proba:
                    best_proba = probabilities[classLabel]
                    best_class_label = classLabel
            result.append(best_class_label)
        return result
