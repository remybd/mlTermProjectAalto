import numpy as np;
from sklearn import linear_model;
from regression.LinearRegression import LinearRegression;
from sklearn.decomposition import PCA
from sklearn.cross_validation import ShuffleSplit


deg_max = 50;
iterations = 1000;


# get training data from csv
data = np.loadtxt("../data/regression_dataset_training.csv",int,'#',',',);
indexVote = len(data[0])-1;
X_data = data[:,range(1,indexVote)];
y_data = data[:,indexVote];



#WHITOUT LIB
handRegressor = LinearRegression()
w = handRegressor.fit(X_data,y_data)
#print(w)

mse = np.mean((handRegressor.predict(X_data) - y_data) ** 2);
print("mse without lib on training data=",mse);



#WITH LIB
reg = linear_model.LinearRegression();
reg.fit(X_data, y_data);
#print(reg.coef_);

#get mean square error
mse = np.mean((reg.predict(X_data) - y_data) ** 2);
print("mse with lib on training data=",mse);



#Final result on test
test_data = np.loadtxt("../data/regression_dataset_testing.csv",int,'#',',',);
X_test = test_data[:,range(1,indexVote)];
testSol_data = np.loadtxt("../data/regression_dataset_testing_solution.csv",int,'#',',',);
y_test = testSol_data[:,1];

mse = np.mean((handRegressor.predict(X_test) - y_test) ** 2);
print("mse without lib on test data=",mse);

mse = np.mean((reg.predict(X_test) - y_test) ** 2);
print("mse with lib ont test data=",mse);



totalFeatures = len(X_data[0])
print("\nfeature selection")

def get_x_y(data, index_vote):
    return data[:, range(1, index_vote)], data[:, index_vote]


def getMseForFeature(nb_features,X_train, y_train, X_val, y_val):
    # feature extraction
    pca = PCA(n_components=nb_features)
    fit = pca.fit(X_train)
    # summarize scores
    features = fit.transform(X_train)

    # WITH LIB
    featureReg = linear_model.LinearRegression();
    featureReg.fit(features, y_train);

    # get mean square error on training set
    mse_train = np.mean((featureReg.predict(features) - y_train) ** 2);

    # get mean square error on test set
    features_test = fit.transform(X_val)
    mse_val = np.mean((featureReg.predict(features_test) - y_val) ** 2);

    return mse_train, mse_val;




def checkAllFeatures( X_train, y_train, X_val, y_val):
    mse_train_by_feature = [0] * totalFeatures
    mse_test_by_feature = [0] * totalFeatures

    # Compute mse error for each feature
    for i in range(1, totalFeatures+1):
        mse_train_by_feature[i - 1], mse_test_by_feature[i - 1] = getMseForFeature(i,X_train, y_train, X_val, y_val);

    return mse_train_by_feature, mse_test_by_feature






sum_train_mse = [0] * totalFeatures
sum_val_mse = [0] * totalFeatures
nb_iter = 100
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

    # Compute mse error for each feature
    train_mse_array, val_mse_array = checkAllFeatures(X_train_data, y_train_data, X_valid_data, y_valid_data)
    sum_train_mse = np.add(sum_train_mse, train_mse_array)
    sum_val_mse = np.add(sum_val_mse, val_mse_array)


mean_train_mse = np.divide(sum_train_mse, nb_iter)
mean_val_mse = np.divide(sum_val_mse, nb_iter)




#Get the best
bestFeatureTrainIndex = np.argmin(mean_val_mse)
bestFeatureTrain = bestFeatureTrainIndex +1;
print("best feature after cv =",bestFeatureTrain)
print("mean mse of cv best feature =",mean_val_mse[bestFeatureTrainIndex])


bestFeatureTrainMse, bestFeatureTestMse = getMseForFeature(bestFeatureTrain,X_data, y_data, X_test, y_test)
print("train mse for best cv feature =",bestFeatureTrainMse)
print("test mse for best cv feature =",bestFeatureTestMse)


training_mse_array, test_mse_array = checkAllFeatures(X_data, y_data, X_test, y_test)
bestFeatureTestIndex = np.argmin(test_mse_array)
bestFeatureTest = bestFeatureTestIndex +1;
print("\nbest feature test=",bestFeatureTest)
print("mse on test =",test_mse_array[bestFeatureTestIndex])


