from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

'''
In general, Decision Trees are not affected by the scale or range of the input features, 
since they only split the data based on the values of the features, and not their absolute magnitudes. 
Therefore, there is usually no need to standardize or normalize the data before training a Decision Tree classifier. 
However, if the dataset has some features that are on a much larger scale than others, it may be a good idea to scale 
the features to ensure that they are all equally important to the model.
In such cases, you can use scikit-learn's StandardScaler or MinMaxScaler classes to scale the features.
'''

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = clf.predict(X_test)

# Computing Metrics
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Plot heatmap of confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Accuracy score
print('accuracy is', accuracy_score(y_pred,y_test))

################################################################################################################
################################### ROC Curve ##################################################################
################################################################################################################

roc_auc_score(y_test, clf.predict_proba(X_test), multi_class = "ovr")

# Predict probabilities for test set
y_score = clf.predict_proba(X_test)[:, 1]

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

# Generate a Graphviz dot file for the tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=iris.feature_names,  
                           class_names=iris.target_names,  
                           filled=True, rounded=True,  
                           special_characters=True)

# Visualize the tree using Graphviz
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")


################################################################################################################
########################################### OPTIMIZATION #######################################################
################################################################################################################

# Define the parameter grid to search
param_grid = {'max_depth': [2, 4, 6, 8],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}


# Create a GridSearchCV object with the Decision Tree classifier and parameter grid
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)  # Output: {'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}
print(best_score)   # Output: 0.9523809523809523


################################################################################################################
########################################## POST PRUNING ########################################################
################################################################################################################

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

dts = []
for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    dt.fit(X_train, y_train)
    dts.append(dt)
    
train_scores = [dt.score(X_train, y_train) for dt in dts]
test_scores = [dt.score(X_test, y_test) for dt in dts]
