'''
Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and 
robustness of the model. It works by building a forest of decision trees on random subsets of the data and features, 
and then aggregating their predictions to make a final prediction.
'''

# Import the necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree._tree import TREE_LEAF


# Load the dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

'''
 Random Forest is also a good choice when you have missing values or outliers in your data. It is not suitable for 
 problems with very few observations, as it can overfit the data in such cases.
 '''

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Random Forest is not sensitive to the scale of the features, so you don't need to standardize or normalize the
 data before applying the model. However, it's still a good practice to scale the data if you have features with 
 different ranges or units, as it can improve the performance of the model and make the training process faster.
 '''
 
# Initialize the Random Forest Classifier with n_estimators=100 and random_state=42
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training set
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Get classification report
report = classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

# Print classification report
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap of confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model's accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

################################################################################################################
################################### ROC Curve ##################################################################
################################################################################################################

# Predict probabilities for test set
y_score = rfc.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

################################################################################################################
################################### Visualization ##############################################################
################################################################################################################

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

export_graphviz(clf, out_file='tree.dot', feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True)

# run different numbers of trees to see the effect of the number on the accuracy of the prediction
n = 100
accuracy = [0]*n

for i in range(n):
    classifier = RandomForestClassifier(n_estimators=i+1)
    classifier = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy[i] = accuracy_score(y_test, predictions)

plt.plot(range(1, n+1), accuracy)
plt.xlabel("Number of trees")
plt.ylabel("Accuracy of prediction")
plt.title("Effect of the number of trees on the prediction accuracy")
plt.show()

################################################################################################################
################################### Optimization ###############################################################
################################################################################################################

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

################################################################################################################
########################################## POST PRUNING ########################################################
################################################################################################################

def prune_index(inner_tree, index, threshold):
    """
    Prunes the decision tree recursively
    """
    if inner_tree.value[TREE_LEAF, 0] > threshold:
        # turn node into leaf by deleting child nodes
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    else:
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_left[index], threshold)
        if inner_tree.children_right[index] != TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_right[index], threshold)

threshold = 0.05
for tree in rfc.estimators_:
    prune_index(tree.tree_, 0, threshold)

y_pred_pruned = rfc.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print('Pruned Accuracy:', accuracy_pruned)
