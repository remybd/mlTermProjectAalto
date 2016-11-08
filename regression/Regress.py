import numpy as np
from sklearn import linear_model;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.kernel_ridge import KernelRidge;
from sklearn.cross_validation import ShuffleSplit


def report_performance(mean_train_mse,mean_val_mse,  classifiers):
    for i in range(0, len(classifiers)):
        print(classifiers[i]['name'])
        print("\tmean mse for training =",mean_train_mse[i])
        print("\tmean mse for validation =",mean_val_mse[i])


def mse(X, y, reg):
    return np.mean((reg.predict(X) - y) ** 2);


def get_errors(classifier, X_train, y_train, X_val, y_val):
    reg = classifier.fit(X_train, y_train)
    mse_train = mse(X_train,y_train,reg)
    mse_val = mse(X_val,y_val,reg)
    return mse_train, mse_val;


def get_x_y(data, index_vote):
    return data[:, range(1, index_vote)], data[:, index_vote]





# Read data
data = np.loadtxt("data/regression_dataset_training.csv", dtype=int, skiprows=1, delimiter=',',)
indexVote = len(data[0])-1
X_data = data[:, range(1, indexVote)]
y_data = data[:, indexVote]

# Setup Classifiers
linear = {'classifier': linear_model.LinearRegression(), 'name': "LinearRegression"}
lasso = {'classifier': linear_model.LassoCV(), 'name': "LassoRegression"}
ridge = {'classifier': linear_model.Ridge(), 'name': "RidgeRegression"}
ridgeCV = {'classifier': linear_model.RidgeCV(), 'name': "RidgeRegressionCV"}
gaussian = {'classifier': GaussianProcessRegressor(), 'name': "GaussianProcessRegression"}
kernel = {'classifier': KernelRidge(), 'name': "KernelRidgeRegression"}

classifiers = [linear, lasso, ridge, ridgeCV, gaussian, kernel]



sum_train_mse = [0] * len(classifiers)
sum_val_mse = [0] * len(classifiers)
nb_iter = 200
# Random repartition for training & validation data in order to perform cross validation
rs = ShuffleSplit(len(data[:, 0]), n_iter=nb_iter, test_size=0.25)
i = 1;
for train_indexes, val_indexes in rs:
    print(i)
    i += 1;
    # Split in validation & training set
    valid_data, train_data = data[val_indexes, :], data[train_indexes, :]
    X_train_data, y_train_data = get_x_y(train_data, indexVote)
    X_valid_data, y_valid_data = get_x_y(valid_data, indexVote)

    # Compute mse error for each classifier
    for classifier_index, item in enumerate(classifiers):
        classifier = classifiers[classifier_index]['classifier']
        mse_train, mse_val = get_errors(classifier, X_train_data, y_train_data, X_valid_data, y_valid_data)
        sum_train_mse[classifier_index] += mse_train
        sum_val_mse[classifier_index] += mse_val


mean_train_mse = np.divide(sum_train_mse, nb_iter)
mean_val_mse = np.divide(sum_val_mse, nb_iter)

report_performance(mean_train_mse, mean_val_mse, classifiers)


#Get the best
index_best_classifier = np.argmin(mean_val_mse)
best_classifier = classifiers[index_best_classifier];
print("\nbest classifier : ", best_classifier['name'])
print("best classifier validation mean mse =", mean_val_mse[index_best_classifier])


'''
# Perform best classifier on test data
'''
test = np.loadtxt("data/regression_dataset_testing.csv",dtype=int, skiprows=1, delimiter=',',);
X_test = test[:,range(1,indexVote)];
testSol = np.loadtxt("data/regression_dataset_testing_solution.csv",dtype=int, skiprows=1, delimiter=',',);
y_test = testSol[:,1];

mse_test = mse(X_test,y_test,best_classifier['classifier'])
print("\nmse on test set =", mse_test);