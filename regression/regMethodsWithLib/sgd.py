import numpy as np;
from sklearn import linear_model;


deg_max = 50;
iterations = 1000;


# get training data from csv
data = np.loadtxt("../data/regression_dataset_training.csv",int,'#',',',);
indexVote = len(data[0])-1;


#WITH LIB
X_data = data[:,range(1,indexVote)];
y_data = data[:,indexVote];

#get coeff with lib
reg = linear_model.SGDRegressor();
reg.fit(X_data, y_data);

#get mean square error
mse = np.mean((reg.predict(X_data) - y_data) ** 2);
print(mse);



#Final result on test
test = np.loadtxt("../data/regression_dataset_testing.csv",int,'#',',',);
X_test = test[:,range(1,indexVote)];
testSol = np.loadtxt("../data/regression_dataset_testing_solution.csv",int,'#',',',);
y_test = testSol[:,1];

mse = np.mean((reg.predict(X_test) - y_test) ** 2);
print(mse);
