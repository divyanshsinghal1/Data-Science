# =============================================================================
# Machine Learning Algorithms
# =============================================================================


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Setting directory
os.chdir("C:/Users/asus/Desktop/TalentShiksha/Scripts/PandasWorkingDirectory")


# =============================================================================
# Reading Data
# =============================================================================
  
data = pd.read_csv("Iris.csv")  


data.shape

data.set_index('Id', inplace = True)
data.reset_index()
# =============================================================================
# Prepare Training and Test
# =============================================================================
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values



# Splitting the data into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# =============================================================================
# Normalization vs Standardization
# =============================================================================
df = pd.read_csv("Big Mart Sales Prediction.csv")

df = df.select_dtypes(include = ["float64", "int64"])

# Notice the scales
df.describe()


# Splitting the data into Train and Test (Consider Item_Sales as a predicting column)
from sklearn.model_selection import train_test_split

X_sales = df.iloc[:, :-1]
y_sales = df.iloc[:, -1]

X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(X_sales, y_sales, test_size = 0.3, random_state = 2)



##
## Normalization (Ranging between 0 and 1, (x- Xmin)/(Xmax - Xmin) using sklearn
##
from sklearn.preprocessing import MinMaxScaler

# Calling scaler 
scaler_norm = MinMaxScaler()

# Fitting scaler on training data
scaler_norm = scaler_norm.fit(X_train_sales)

# Transform Training Data
X_train_sales_norm = scaler_norm.transform(X_train_sales) 
X_train_sales_norm[:,1].min()


X_train_sales_norm = scaler_norm.fit_transform(X_train_sales) 



# Transform testing values
X_test_sales_norm = scaler_norm.transform(X_test_sales)



##
## Standardization (Centering around mean with unit standard deviation), (X - mean)/sigma
##
from sklearn.preprocessing import StandardScaler


# Calling Standardizer
scaler_stand = StandardScaler()

# Fitting standardizer on training data
scaler_stand = scaler_stand.fit(X_train_sales)


# Transform Training Data
X_train_sales_stand = scaler_stand.transform(X_train_sales) 

# Transform testing values
X_test_sales_stand = scaler_norm.transform(X_test_sales)


Note: One hot encoded variable should only be applied with Min Max Scaler and not Standard Scaler at all since Standardisation would assign distribution to categorical features which is not desirable


# =============================================================================
# Encoding
# =============================================================================
Typically, any structured dataset includes multiple columns – a combination of numerical as well as categorical variables. A machine can only understand the numbers. It cannot understand the text. That’s essentially the case with Machine Learning algorithms too.

That’s primarily the reason we need to convert categorical columns to numerical columns so that a machine learning algorithm understands it. This process is called categorical encoding.


##
## Label Encoding
##

In label Encoding Technique, a unique integer is assigned a unique integer based on alphabetical odering. 

df_sal = pd.read_csv("SalaryData.csv")

# Import label encoder
from sklearn import preprocessing

# Initializing
label_encoder = preprocessing.LabelEncoder()

# Encoding labels of Country column
df_sal["Country1"] = label_encoder.fit_transform(df_sal["Country"])


Challenges with label encoding are it creates ranking based on alphabets, due to this model will capture the relationship between countries such India < Japan < US due to starting alphabets. 


##
## One hot Encoding
##
This technique simply creates additional features based on the number of unique values in the categorical feature. Every unique value in the category will be added as a feature.Here, each category is represented as a one-hot vector

from sklearn import preprocessing

# Reading Data
df_sal = pd.read_csv("SalaryData.csv")


# Generating binary variables using get_dummies
df_sal_dum = pd.get_dummies(df_sal, columns = ["Country"], drop_first=True) #drop_first=True


We use drop_first = True in order to avoid from Dummy Variable Trap situation, Country_First + Country_Germany + Country_Spain = 1



We apply One-Hot Encoding when:

a) The categorical feature is not ordinal (like the countries above)
b) The number of categorical features is less so one-hot encoding can be effectively applied

We apply Label Encoding when:

a) The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
b) The number of categories is quite large as one-hot encoding can lead to high memory consumption
    
    
    


# =============================================================================
# Imbalanced Class Dataset Handling
# =============================================================================


df_im = pd.read_csv("Dataset - Imbalanced.csv")


df_im["ChurnFlag"]
plot = sns.countplot(df_im["ChurnFlag"])
plot.set_xticklabels(["Not Churn", "Churn"])

df_im["ChurnFlag"].value_counts()


# Counting classes
class_count_0, class_count_1 = df_im["ChurnFlag"].value_counts()

# Seperate class
class_0 = df_im[df_im["ChurnFlag"] == 0]
class_1 = df_im[df_im["ChurnFlag"] == 1]


##
## Random Under Sampling (Removing some observations of the majority class)
##

# Downsampling majority class data
class_0_under = class_0.sample(class_count_1, random_state=1)


# Combining final dataset
df_im_under = pd.concat([class_0_under, class_1], axis = "rows")


df_im_under['ChurnFlag'].value_counts().plot(kind='bar', title='count (target)')


##
## Random Over-sampling (Rows are repeated)
##

class_1_over = class_1.sample(class_count_0, replace = True, random_state = 1)


# Combining final dataset
df_im_over = pd.concat([class_1_over, class_0], axis = "rows")


df_im_over['ChurnFlag'].value_counts().plot(kind='bar', title='count (target)')


##
## Random Under-Sampling Using Imblearn 
##

# Forms cluster of data first and then removes records from clusters preserving information overall. More effective than above.

# pip install imblearn
from imblearn.under_sampling import RandomUnderSampler

us_imblearn = RandomUnderSampler(random_state=1, replacement = True)

x_ros, y_ros = us_imblearn.fit_sample(df_im.iloc[:,0:-1], df_im.iloc[:,-1])

# Combining final dataset
df_im_under = pd.concat([x_ros, y_ros], axis = "columns")



##
## Randome Over-Sampling Using Imblearn
##

# import library
from imblearn.over_sampling import RandomOverSampler

os_imblearn = RandomOverSampler(random_state=42)


# fit predictor and target variable
x_ros, y_ros = os_imblearn.fit_sample(df_im.iloc[:,0:-1], df_im.iloc[:,-1])


# Combining final dataset
df_im_over = pd.concat([x_ros, y_ros], axis = "columns")
df_im_over["ChurnFlag"].value_counts()


##
## SMOTE (Synthetic Minority Oversampling Technique)
## 

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

X, y = make_classification(n_samples=10000, n_features = 2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

#X, y = make_classification(n_samples=200, random_state=0, n_features=10)

#test = pd.DataFrame(X, y)

Counter(y)


# Calling
smote = SMOTE(k_neighbors = 3)

# Fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X, y)

Counter(y_smote)



# =============================================================================
# Linear Regression
# =============================================================================
data = pd.read_csv("Iris.csv")  

data.shape


# =============================================================================
# Prepare Training and Test
# =============================================================================
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting the data into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# Label Encoder - Converts categorical variable to numerical variable by assigning numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)  # this is Y_train categorical to numerical

# This is only for Linear Regression 
X_trainL, X_testL, y_trainL, y_testL = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Importing linear Regression
from sklearn.linear_model import LinearRegression

# Calling Linear Regression Class
modelLR = LinearRegression()

# Fitting the model
modelLR.fit(X_trainL, y_trainL)


# Predicting on test data
Y_pred = modelLR.predict(X_testL)


# Computing Metrics
from sklearn import metrics


print('y-intercept             :' , modelLR.intercept_)
print('beta coefficients       :' , modelLR.coef_)

print('Mean Abs Error MAE      :', metrics.mean_absolute_error(y_testL,Y_pred))

print('Mean Squared Error MSE     :', metrics.mean_squared_error(y_testL,Y_pred))

print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_testL,Y_pred)))

print('r2 value                :', metrics.r2_score(y_testL,Y_pred))

from sklearn import metrics

# Predicting on test data
Y_pred = modelLR.predict(X_test)
metrics.r2_score(y_test, y_pred)


# Adjusted R Square on training data
print('adjusted r2 value                :' , 1 - (1-modelLR.score(X_trainL, y_trainL))*(len(y_trainL)-1)/(len(y_trainL)-X_trainL.shape[1]-1))





# =============================================================================
# Decision Tree
# =============================================================================
X_train.drop('Id', axis = 'columns', inplace = True)
X_test.drop('Id', axis = 'columns', inplace = True)

from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

model_dt = DecisionTreeClassifier(criterion='entropy')

#plt.figure(figsize=(18, 12))
#plot_tree(model_dt)



model_dt.fit(X_train, y_train)

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(model_dt,
               feature_names = fn, 
               class_names=cn,
               filled = True);

model_dt.feature_importances_


y_pred = model_dt.predict(X_test)

y_pred_prob = model_dt.predict_proba(X_test)



# Summary Predictions
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



# Accuracy score
print('accuracy is', accuracy_score(y_pred,y_test))

# ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, model_dt.predict_proba(X_test), multi_class = "ovr")


# =============================================================================
# Random Forest
# =============================================================================
X_train.drop("Id", axis = 'columns', inplace = True)
X_test.drop("Id", axis = 'columns', inplace = True)


from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestClassifier(n_estimators=100, criterion='entropy')

model_rf.fit(X_train, y_train)

model_rf.estimators_
model_rf.feature_importances_



y_pred = model_rf.predict(X_test)
y_pred_prob = model_rf.predict_proba(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

roc_auc_score(y_test, model_rf.predict_proba(X_test), multi_class = "ovr")



# =============================================================================
# Logistics Regression
# =============================================================================

# It predicts the probability of occurrence of an event by fitting data to a logit function. Hence, it is also known as logit regression. Since, it predicts the probability, its output values lies between 0 and 1

from sklearn.linear_model import LogisticRegression

model_ltr = LogisticRegression(max_iter=1000, random_state=123)
model_ltr.fit(X_train, y_train)

y_pred = model_ltr.predict_proba(X_test) # predicts probability
y_pred = model_ltr.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))



# =============================================================================
# K Nearest Neighbours
# =============================================================================
Widely used in classification problem, being simple algo it stores all available cases and classifies new cases by a majority votes if its k neighbours. 

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=8, metric = 'euclidean') # Default 8

model_knn.fit(X_train, y_train) 

y_pred = model_knn.predict(X_test)
y_pred = model_knn.predict_proba(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

 
# =============================================================================
# Naive Bayes    
# =============================================================================
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

y_pred = model_nb.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))





# =============================================================================
# Support Vector Machines
# =============================================================================
from sklearn.svm import SVC #(Support Vector Classifier)
#from sklearn.svm import SVR 

model_svc = SVC(C = 1.0, kernel = 'rbf', random_state=123)
model_svc.fit(X_train, y_train)

y_pred = model_svc.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

    
# =============================================================================
# Extra Tree Classifier
# =============================================================================
from sklearn.tree import ExtraTreeClassifier

model_etc = ExtraTreeClassifier()

model_etc.fit(X_train, y_train)

y_pred = model_etc.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# =============================================================================
# Bagging Classifier
# =============================================================================
from sklearn.ensemble import BaggingClassifier

model_bc = BaggingClassifier()
model_bc.fit(X_train,y_train)
y_pred = model_bc.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))

#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# =============================================================================
# Linear Discriminant Analysis
# =============================================================================
# =============================================================================
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 
# model_lda =LinearDiscriminantAnalysis()
# 
# model_lda.fit(X_train,y_train)
# 
# y_pred=model_lda.predict(X_test)
# 
# # Summary of the predictions made by the classifier
# print(classification_report(y_test,y_pred))
# 
# print(confusion_matrix(y_pred,y_test))
# 
# #Accuracy Score
# print('accuracy is ',accuracy_score(y_pred,y_test))
# =============================================================================


# =============================================================================
# K Means Clustering Algorithm
# =============================================================================
x = data.iloc[:, [1, 2, 3, 4]].values

# Elbow Method - For finding the best number of clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = [] # Within cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, random_state = 0, verbose = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # within cluster sum of squares
plt.show()


# Fitting K Means model
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

kmeans.fit(x)
kmeans.predict(x)


y_kmeans = kmeans.fit_predict(x)



# Visualizing the Cluster
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-Setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-Versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'Iris-Virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'green', label = 'Centroids',marker='*')

plt.legend()

test = pd.DataFrame(x, y_kmeans)
test.reset_index(inplace = True)

test.columns = ['Cluster', 0 ,1, 2, 3]

# =============================================================================
# Boosting Algorithms
# =============================================================================

##
## AdaBoost Classifier
##
from sklearn.ensemble import AdaBoostClassifier

model_adc = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=123)

model_adc.fit(X_train,y_train)

y_pred=model_adc.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_pred,y_test))

#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))



##
## XGBoost
##
from xgboost import XGBClassifier, plot_tree

# Dataset
X = np.array([2,8,12,18]).reshape((4,1))
y = np.array([0,1,1,0])

# Define parameters and fit XgBoost Model
model=XGBClassifier(max_depth=2,learning_rate=0.8,n_estimators=2,gamma=2,
                    min_child_weight=0,reg_alpha=0,reg_lambda=0,base_score=0.5)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print('accuracy is ',accuracy_score(y_pred,y_test))



##
## Gradient Boosting Classifier
##
from sklearn.ensemble import GradientBoostingClassifier


model_gbc = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=100)
model_gbc.fit(X_train,y_train)
y_pred=model_gbc.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_pred,y_test))

#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))




# =============================================================================
# Principal Component Analysis
# =============================================================================
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
#70,000 images with 784 dimensions (784 features)

tmp = pd.DataFrame(mnist.data)
mnist.data.shape

mnist.data


mnist.target
mnist.target.shape


from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


train_img.shape

# Standardizing Data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


from sklearn.decomposition import PCA

# Make an instance of the Model
pca = PCA(n_components=0.95) # Form components such that 95% of the variance is retained 


# Fiting on training data
pca.fit(train_img)

pca.components_ # Components data
2

# Number of components selected
len(pca.components_)

# Transforming on Test Data
train_img1 = pca.transform(train_img)
train_img1.shape 

test_img1 = pca.transform(test_img)
train_img1.shape 


# Variation Explained by PCA
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)


cum_var_explained = np.cumsum(percentage_var_explained)


# Plot the PCA spectrum
plt.figure(1, figsize = (12, 6))
plt.plot(cum_var_explained, linewidth = 2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


pca.transform(train_img, 150)



# =============================================================================
# Grid Search with Cross Validation
# =============================================================================

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 200, 300]
}


2 * 4 * 2 * 3 * 3 * 4

# Create a based model
rf = RandomForestClassifier()

X, y = make_classification(n_samples=200, random_state=0)


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)


# 1 * 4 * 2 * 3 * 3 * 4 = 288 Combinations

# Fit the grid search to the data
grid_search.fit(X, y)

grid_search.best_params_
grid_search.cv_results_["mean_test_score"]
grid_search.cv_results_["std_test_score"]

rf = grid_search.best_estimator_

rf.fit(X, y)

rf.predict()

# =============================================================================
# Random Search Using Cross Validation
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# Use the random grid to search for best hyperparameters

# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 3, verbose=1, random_state=42, n_jobs = -1)


# Fit the random search model
rf_random.fit(X, y)


rf_random.best_params_
{'bootstrap': True,
 'max_depth': 70,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 400}






























