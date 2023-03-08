'''
Extra Trees Classifier is similar to the Random Forest algorithm. It builds multiple decision trees using a random subset of 
the features and data, then combines their predictions to make the final classification decision. However, unlike Random Forest,
 Extra Trees Classifier selects the split points for each node in the decision tree randomly. This means that each decision 
 tree in Extra Trees Classifier is trained on a different subset of the data and features, and the split points are chosen 
 randomly without regard to the quality of the split. This makes Extra Trees Classifier less prone to overfitting and more 
 robust to noisy data than Random Forest.

During training, the Extra Trees Classifier algorithm selects a random subset of the data and features and builds a decision 
tree using the selected subset. For each node in the tree, the algorithm randomly selects a subset of the features and a split 
point for each feature. It then selects the feature and split point that produces the highest decrease in impurity 
(e.g., Gini impurity or entropy) as the best split for the node. The algorithm continues this process recursively until 
it reaches a leaf node or a stopping criterion is met.

During prediction, the Extra Trees Classifier algorithm aggregates the predictions of all decision trees to make the final 
classification decision. Each decision tree casts a vote for the class of the input sample, and the class with the most votes 
is chosen as the final prediction.

################### Advantages of Extra Trees Classifier

Extra Trees Classifier is less prone to overfitting than other decision tree-based algorithms, such as decision trees and 
Random Forest, because it uses randomized splitting rules and selects a subset of the data and features for each tree.

Extra Trees Classifier is faster to train and predict than Random Forest because it requires less time to select the best split points.

Extra Trees Classifier is robust to noisy data because it uses a randomized splitting rule that reduces the impact of noisy 
data on the decision tree.

################### Disadvantages of Extra Trees Classifier

Extra Trees Classifier may produce biased results if the input features have different scales or distributions because it 
uses a randomized splitting rule that is sensitive to the scale and distribution of the input features.

Extra Trees Classifier may require more trees than Random Forest to achieve high accuracy because it uses a random splitting 
rule that produces less accurate splits than the optimal splitting rule.
'''

from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load iris dataset
iris = load_iris()

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# create an Extra Trees Classifier model
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# train the model on the training data
et_model.fit(X_train, y_train)

# evaluate the model on the testing data
y_pred = et_model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))


