from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

# Load the Boston Housing dataset
X, y = load_boston(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Ridge regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict on the test set
y_pred = ridge.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)


##############################################################################
##############################################################################


# Fit a Ridge regression model for a range of alpha values
alphas = np.logspace(-4, 0, 100)
coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

# Plot the Ridge coefficient plot
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


##############################################################################
##############################################################################

from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset
X, y = load_boston(return_X_y=True)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a RidgeCV object
alphas = np.logspace(-6, 6, 200)
ridge_cv = RidgeCV(alphas=alphas, cv=5)

# Fit the RidgeCV object to the data
ridge_cv.fit(X, y)

# Get the coefficients for each alpha
coefs = []
for a in ridge_cv.alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# Plot the coefficient progression
ax = plt.gca()
ax.plot(ridge_cv.alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficient progression')
plt.axis('tight')
plt.show()

##############################################################################
##############################################################################

# Fit a Ridge regression model with a chosen alpha value
alpha = 0.1
ridge = Ridge(alpha=alpha, fit_intercept=False)
ridge.fit(X_train, y_train)

# Calculate the predicted values and residuals on the test set
y_pred = ridge.predict(X_test)
residuals = y_test - y_pred

# Plot the Ridge residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.title('Ridge residual plot')
plt.show()