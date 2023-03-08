'''
Bagging Classifier
Bagging (Bootstrap Aggregating) is a machine learning ensemble algorithm that uses bootstrap sampling to create 
multiple subsets of the original dataset, which are then used to train multiple models. Each model is trained on 
a different subset of the data and uses a different random seed, resulting in a diverse set of models. The outputs 
of these models are then aggregated to make a final prediction.

The Bagging Classifier in scikit-learn is an implementation of the Bagging ensemble algorithm for classification problems.
 It can be used with different base classifiers and can be tuned to optimize its performance.
 
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib as plt

# Load the dataset
X, y = load_dataset()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the base classifier to use
base_classifier = DecisionTreeClassifier()

# Create a bagging classifier with 10 base estimators
bagging = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=0)

# Train the bagging classifier on the training set
bagging.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = bagging.predict(X_test)

# Measure the accuracy of the bagging classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))

feature_importances = base_classifier.feature_importances_
oob_score = bagging.oob_score_
proba = bagging.predict_proba(X_test)

################################################################################################################
########################################### OPTIMIZATION #######################################################
################################################################################################################

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [5, 10, 15],
    'base_estimator': [DecisionTreeClassifier(), RandomForestClassifier()],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
    'bootstrap': [True, False],
    'random_state': [0, 1, 2]
}

# Create a grid search object and fit it to the training data
grid_search = GridSearchCV(BaggingClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print(f'Best parameters: {grid_search.best_params_}')

################################################################################################################
################################### Visualization ##############################################################
################################################################################################################

# Visualize the decision boundaries

plot_decision_regions(X_test, y_test, clf=bagging)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries')
plt.show()
