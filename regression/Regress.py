import numpy as np
from sklearn import linear_model;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.kernel_ridge import KernelRidge;
from sklearn.cross_validation import ShuffleSplit
from regression.LinearRegression import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2




def report_performance(train_best_mse_by_reg,val_best_mse_by_reg,  regressors):
    print(train_best_mse_by_reg)
    print(val_best_mse_by_reg)
    for reg_index, item in enumerate(regressors):
        print(regressors[reg_index]['name'])
        print("\tmean mse for training =",train_best_mse_by_reg[reg_index][1])
        print("\tmean mse for validation =",val_best_mse_by_reg[reg_index][1])
        print("\tbest number of features on training =", train_best_mse_by_reg[reg_index][0])
        print("\tbest number of features on validation =", val_best_mse_by_reg[reg_index][0])


def mse(X, y, reg):
    return np.mean((reg.predict(X) - y) ** 2);


def regress(regressor, X_train, y_train, X_val, y_val):
    regressor.fit(X_train, y_train)
    mse_train = mse(X_train,y_train,regressor)
    mse_val = mse(X_val,y_val,regressor)
    return mse_train, mse_val;



def getMseForFeature(regressor, nb_features,X_train, y_train, X_val, y_val):
    # feature extraction
    kbest = SelectKBest(score_func=chi2, k=nb_features)
    fit = kbest.fit(X_train,y_train)

    # summarize scores
    features_train = fit.transform(X_train)
    features_val = fit.transform(X_val)

    mse_train, mse_val = regress(regressor,features_train, y_train, features_val, y_val)

    return mse_train, mse_val;




def regressAllFeatures(regressor, X_train, y_train, X_val, y_val):
    totalFeatures = len(X_train[0])
    mse_train_by_feature = [0] * totalFeatures
    mse_test_by_feature = [0] * totalFeatures

    # Compute mse error for each feature on train and validation
    for i in range(1, totalFeatures + 1):
        mse_train_by_feature[i - 1], mse_test_by_feature[i - 1] = getMseForFeature(regressor, i, X_train, y_train, X_val, y_val);

    return mse_train_by_feature, mse_test_by_feature




def regressAllRegressor(regressors, X_train, y_train, X_val, y_val):
    train_mse_by_reg_by_feat = [0] * len(regressors)
    val_mse_by_reg_by_feat = [0] * len(regressors)

    # Compute mse error for each regressor
    for reg_index, item in enumerate(regressors):
        regressor = regressors[reg_index]['regressor']

        # Compute mse error for each feature
        train_mse_by_reg_by_feat[reg_index], val_mse_by_reg_by_feat[reg_index] = regressAllFeatures(regressor,X_train_data, y_train_data, X_valid_data, y_valid_data)

    return train_mse_by_reg_by_feat, val_mse_by_reg_by_feat



def get_x_y(data, index_vote):
    return data[:, range(1, index_vote)], data[:, index_vote]


def getBestFeatureByRegressor(regressors, mse_by_reg_by_feat):
    best_mse_by_reg = [0] * len(regressors)

    # get best mse error for each regressor
    for reg_index, item in enumerate(regressors):
        bestFeatureIndex = np.argmin(mse_by_reg_by_feat[reg_index])
        bestFeatureNumber = bestFeatureIndex + 1;
        bestMse = mse_by_reg_by_feat[reg_index][bestFeatureIndex]

        best_mse_by_reg[reg_index] = [bestFeatureNumber,bestMse]

    return best_mse_by_reg




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

regressors = [linear, lasso, ridgeCV]


sum_train_mse_by_reg_by_feat = [[0] * len(X_data[0])]* len(regressors)
sum_val_mse_by_reg_by_feat = [[0] * len(X_data[0])]* len(regressors)
nb_iter = 1
# Random repartition for training & validation data in order to perform cross validation
rs = ShuffleSplit(len(data[:, 0]), n_iter=nb_iter, test_size=0.25)
i = 1

for train_indexes, val_indexes in rs:
    # Split in validation & training set
    valid_data, train_data = data[val_indexes, :], data[train_indexes, :]
    X_train_data, y_train_data = get_x_y(train_data, indexVote)
    X_valid_data, y_valid_data = get_x_y(valid_data, indexVote)

    # Compute mse error for each regressor
    train_mse_by_reg_by_feat, val_mse_by_reg_by_feat = regressAllRegressor(regressors, X_train_data, y_train_data, X_valid_data, y_valid_data)

    sum_train_mse_by_reg_by_feat = np.add(sum_train_mse_by_reg_by_feat, train_mse_by_reg_by_feat)
    sum_val_mse_by_reg_by_feat = np.add(sum_val_mse_by_reg_by_feat, val_mse_by_reg_by_feat)

    print(i)
    i += 1;


mean_train_mse_by_reg_by_feat = np.divide(sum_train_mse_by_reg_by_feat, nb_iter)
mean_val_mse_by_reg_by_feat = np.divide(sum_val_mse_by_reg_by_feat, nb_iter)

train_best_mse_by_reg = getBestFeatureByRegressor(regressors,mean_train_mse_by_reg_by_feat)
val_best_mse_by_reg = getBestFeatureByRegressor(regressors,mean_val_mse_by_reg_by_feat)

report_performance(train_best_mse_by_reg, val_best_mse_by_reg, regressors)


#Get the best
index_best_regressor = np.argmin(val_best_mse_by_reg[:][1])
best_regressor = regressors[index_best_regressor];
nb_features_best_reg = val_best_mse_by_reg[index_best_regressor][0]
print("\nbest regressor : ", best_regressor['name'])
print("\nnb of features : ", nb_features_best_reg)
print("best regressor validation mean mse =", val_best_mse_by_reg[index_best_regressor][1])


'''
# Perform best regressor on test data
'''
test = np.loadtxt("data/regression_dataset_testing.csv",dtype=int, skiprows=1, delimiter=',',);
X_test = test[:,range(1,indexVote)];
testSol = np.loadtxt("data/regression_dataset_testing_solution.csv",dtype=int, skiprows=1, delimiter=',',);
y_test = testSol[:,1];

mse_training, mse_test = getMseForFeature(best_regressor['regressor'],nb_features_best_reg, X_data, y_data, X_test, y_test)
print("\nmse on all training set =", mse_training);
print("mse on test set =", mse_test);