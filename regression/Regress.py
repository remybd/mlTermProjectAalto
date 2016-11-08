import numpy as np
from sklearn import linear_model;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.kernel_ridge import KernelRidge;
from sklearn.cross_validation import ShuffleSplit
from regression.LinearRegression import LinearRegression



def report_performance(mean_train_mse,mean_val_mse,  regressors):
    for i in range(0, len(regressors)):
        print(regressors[i]['name'])
        print("\tmean mse for training =",mean_train_mse[i])
        print("\tmean mse for validation =",mean_val_mse[i])


def mse(X, y, reg):
    return np.mean((reg.predict(X) - y) ** 2);


def get_errors(regressor, X_train, y_train, X_val, y_val):
    regressor.fit(X_train, y_train)
    mse_train = mse(X_train,y_train,regressor)
    mse_val = mse(X_val,y_val,regressor)
    return mse_train, mse_val;


def get_x_y(data, index_vote):
    return data[:, range(1, index_vote)], data[:, index_vote]





# Read data
data = np.loadtxt("data/regression_dataset_training.csv", dtype=int, skiprows=1, delimiter=',',)
indexVote = len(data[0])-1
X_data = data[:, range(1, indexVote)]
y_data = data[:, indexVote]

# Setup Classifiers
linear = {'regressor': linear_model.LinearRegression(), 'name': "LinearRegression"}
linearHand = {'regressor': LinearRegression(), 'name': "LinearHandMadeRegression"}
lasso = {'regressor': linear_model.LassoCV(), 'name': "LassoRegression"}
ridge = {'regressor': linear_model.Ridge(), 'name': "RidgeRegression"}
ridgeCV = {'regressor': linear_model.RidgeCV(), 'name': "RidgeRegressionCV"}
gaussian = {'regressor': GaussianProcessRegressor(), 'name': "GaussianProcessRegression"}
kernel = {'regressor': KernelRidge(), 'name': "KernelRidgeRegression"}

regressors = [linear, linearHand, lasso, ridge, ridgeCV, gaussian, kernel]



sum_train_mse = [0] * len(regressors)
sum_val_mse = [0] * len(regressors)
nb_iter = 10
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

    # Compute mse error for each regressor
    for regressor_index, item in enumerate(regressors):
        regressor = regressors[regressor_index]['regressor']
        mse_train, mse_val = get_errors(regressor, X_train_data, y_train_data, X_valid_data, y_valid_data)
        sum_train_mse[regressor_index] += mse_train
        sum_val_mse[regressor_index] += mse_val


mean_train_mse = np.divide(sum_train_mse, nb_iter)
mean_val_mse = np.divide(sum_val_mse, nb_iter)

report_performance(mean_train_mse, mean_val_mse, regressors)


#Get the best
index_best_regressor = np.argmin(mean_val_mse)
best_regressor = regressors[index_best_regressor];
print("\nbest regressor : ", best_regressor['name'])
print("best regressor validation mean mse =", mean_val_mse[index_best_regressor])


'''
# Perform best regressor on test data
'''
#fit on all the training set
best_regressor["regressor"].fit(X_data, y_data)
mse_training = mse(X_data,y_data,best_regressor['regressor'])
print("\nmse on all trainin set =", mse_training);

#test on data set
test = np.loadtxt("data/regression_dataset_testing.csv",dtype=int, skiprows=1, delimiter=',',);
X_test = test[:,range(1,indexVote)];
testSol = np.loadtxt("data/regression_dataset_testing_solution.csv",dtype=int, skiprows=1, delimiter=',',);
y_test = testSol[:,1];

mse_test = mse(X_test,y_test,best_regressor['regressor'])
print("\nmse on test set =", mse_test);