# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Split data into features and target
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Logistic Regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Get classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Classification report:\n", report)

################################################################################################################
################################### ROC Curve ##################################################################
################################################################################################################

# Plot ROC curve
y_pred_proba = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

################################################################################################################
########################################### OPTIMIZATION #######################################################
################################################################################################################

params = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10]
}

# Perform Grid Search to find best hyperparameters
lr = LogisticRegression(random_state=42)
grid_search = GridSearchCV(lr, params, cv=5)
grid_search.fit(X_train, y_train)

# Predict on test set using best model
y_pred = grid_search.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)