import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset
X, y = load_boston(return_X_y=True)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)

# specify the lasso regression model
model = LassoLarsCV(cv=10, precompute=False).fit(X_train, y_train)

print('Predictors and their regression coefficients:')
d = dict(zip(X.columns, model.coef_))
for k in d:
    print(k, ':', d[k])

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
# ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
print('\nAlpha:', model.alpha_)
plt.axvline(-np.log10(model.alpha_), linestyle="dashed", color='k', label='alpha CV')
plt.ylabel("Regression coefficients")
plt.xlabel("-log(alpha)")
plt.title('Regression coefficients progression for Lasso paths')
plt.show()

# plot mean squared error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.show()

# Mean squared error from training and test data
train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))
print('\nMean squared error for training data:', train_error)
print('Mean squared error for test data:', test_error)

rsquared_train = model.score(X_train, y_train)
rsquared_test = model.score(X_test, y_test)
print('\nR-square for training data:', rsquared_train)
print('R-square for test data:', rsquared_test)