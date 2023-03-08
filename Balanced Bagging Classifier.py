'''
BalancedBaggingClassifier is a machine learning ensemble algorithm that combines a balanced sampling strategy with bagging. 
It is a modified version of the popular Bagging Classifier that is designed to deal with imbalanced datasets.

In a standard bagging ensemble, multiple models are trained on different subsamples of the dataset, with replacement. 
Each model in the ensemble is given equal weight when making predictions, which are then aggregated to produce the final output.
 This can be effective in reducing overfitting and improving model accuracy.

However, when dealing with imbalanced datasets, bagging may not be effective because it can still result in models 
that are biased towards the majority class, which can lead to poor performance on the minority class. 
Balanced Bagging Classifier addresses this issue by using a balanced sampling strategy when creating the subsamples, 
which ensures that the subsamples contain equal numbers of instances from each class.

In this way, Balanced Bagging Classifier creates a set of balanced training datasets and trains a set of base estimators on them, 
and at prediction time, aggregates the results of base estimators to produce a final prediction using a majority 
voting approach. The aim is to improve the classification accuracy on imbalanced datasets, 
by balancing the class distribution and therefore avoiding a bias towards the majority class.
'''

from sklearn.datasets import make_classification
from sklearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Generate an imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9], n_informative=3,
                           n_redundant=1, flip_y=0, n_features=20,
                           n_clusters_per_class=1, n_samples=1000,
                           random_state=10)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                 sampling_strategy='auto',
                                 replacement=False,
                                 random_state=0)

# Fit the model
bbc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bbc.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
