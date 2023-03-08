'''
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm that can be used for both classification and 
regression tasks. It is a simple and intuitive algorithm that works by finding the k-nearest data points to a given 
point and predicting its class (for classification) or value (for regression) based on the majority class or mean value 
of those nearest neighbors.

In KNN, the choice of k determines the number of nearest neighbors to consider when making a prediction.
 A larger value of k can help to reduce the noise in the data but may also result in underfitting, while a smaller 
 value of k may result in overfitting to the training data.
 '''

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix

# Loading data
irisData = load_iris()
  
# Create feature and target arrays
X = irisData.data
y = irisData.target
  
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)

# Create KNN classifier object
k = 5 # choose the number of nearest neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
      
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show() 