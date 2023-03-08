####################################################################################################################
########################################### K Fold Cross Validation ################################################
####################################################################################################################

'''
Cross-validation is a statistical technique used to evaluate machine learning models by dividing the data into 
two sets: training and testing sets. The idea behind cross-validation is to use the training set to train the model 
and the testing set to evaluate its performance.

There are different types of cross-validation techniques, but the most common one is k-fold cross-validation. 
In k-fold cross-validation, the data is divided into k equal-sized subsets. The model is then trained on k-1 subsets 
and tested on the remaining subset. This process is repeated k times, with each subset used once as the testing set.
'''

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# load iris dataset
iris = load_iris()

# create a logistic regression model
model = LogisticRegression()

# set the number of folds for cross-validation
k = 5

# create a k-fold cross-validation object
kf = KFold(n_splits=k)

# create lists to store the accuracy scores
train_accuracy = []
test_accuracy = []

# iterate through each fold
for train_index, test_index in kf.split(iris.data):
    
    # split the data into training and testing sets
    X_train, X_test = iris.data[train_index], iris.data[test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
    
    # train the model on the training set
    model.fit(X_train, y_train)
    
    # evaluate the model on the training set
    train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))
    
    # evaluate the model on the testing set
    test_accuracy.append(accuracy_score(y_test, model.predict(X_test)))

# print the average accuracy scores for each set
print(f"Training Accuracy: {sum(train_accuracy)/k:.2f}")
print(f"Testing Accuracy: {sum(test_accuracy)/k:.2f}")

####################################################################################################################
########################################### Grid Search with Cross Validation ######################################
####################################################################################################################

'''
Grid search with cross-validation is a technique used to tune the hyperparameters of a machine learning model by 
searching over a range of hyperparameters and evaluating the model's performance using cross-validation. 
The idea behind this technique is to find the set of hyperparameters that produces the best performance on the testing set.
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# load iris dataset
iris = load_iris()

# create a support vector machine model
model = SVC()

# set the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# set the number of folds for cross-validation
k = 5

# create a grid search cross-validation object
grid_search = GridSearchCV(model, param_grid, cv=k)

# fit the grid search object to the data
grid_search.fit(iris.data, iris.target)

# print the best parameters and their corresponding score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.2f}")


####################################################################################################################
########################################### Random Search Using Cross Validation ###################################
####################################################################################################################

'''
Random search with cross-validation is another hyperparameter tuning technique that can be used to search for the 
best hyperparameters of a machine learning model. The idea behind this technique is similar to grid search with 
cross-validation, but instead of searching over a pre-defined grid of hyperparameters, we randomly sample from a 
distribution of hyperparameters and evaluate the model's performance using cross-validation.
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform

# load iris dataset
iris = load_iris()

# create a support vector machine model
model = SVC()

# set the parameter distribution for random search
param_dist = {
    'C': uniform(0.1, 100),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] + list(uniform(0.1, 10, 5))
}

# set the number of iterations for random search
n_iter = 50

# set the number of folds for cross-validation
k = 5

# create a random search cross-validation object
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=k)

# fit the random search object to the data
random_search.fit(iris.data, iris.target)

# print the best parameters and their corresponding score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.2f}")
