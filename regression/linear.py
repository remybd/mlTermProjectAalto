import numpy as np;
from sklearn import linear_model;
from regression.LinearRegression import LinearRegression;


deg_max = 50;
iterations = 1000;


# get training data from csv
data = np.loadtxt("data/regression_dataset_training.csv",int,'#',',',);
indexVote = len(data[0])-1;
X_data = data[:,range(1,indexVote)];
y_data = data[:,indexVote];


#WHITOUT LIB
handRegressor = LinearRegression()
w = handRegressor.fit(X_data,y_data)
print(w)

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
test = np.loadtxt("data/regression_dataset_testing.csv",int,'#',',',);
X_test = test[:,range(1,indexVote)];
testSol = np.loadtxt("data/regression_dataset_testing_solution.csv",int,'#',',',);
y_test = testSol[:,1];

mse = np.mean((handRegressor.predict(X_test) - y_test) ** 2);
print("mse without lib on test data=",mse);

mse = np.mean((reg.predict(X_test) - y_test) ** 2);
print("mse with lib ont test data=",mse);
