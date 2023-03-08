###################################################################################################################
############################## SMOTE (Synthetic Minority Oversampling Technique) ##################################
###################################################################################################################

'''
Synthetic Minority Oversampling Technique (SMOTE) is a widely used data augmentation technique for imbalanced datasets in
 machine learning. It is used to address the problem of imbalanced classes, where the number of samples in the minority 
 class is much smaller than that in the majority class.

The main idea behind SMOTE is to generate synthetic samples of the minority class by interpolating between existing samples. 
SMOTE works by randomly selecting a minority class sample and creating synthetic samples along the line segments connecting 
the minority class sample to its k nearest neighbors. The number of synthetic samples generated for each minority class 
sample is controlled by a user-defined parameter called the "sampling ratio" or "oversampling rate".

SMOTE has several advantages over other oversampling techniques such as random oversampling and duplication. 
First, SMOTE does not duplicate existing minority class samples, which may lead to overfitting. 
Second, SMOTE can generate diverse synthetic samples by introducing small variations between samples. 
Third, SMOTE can be easily applied to high-dimensional data without the curse of dimensionality, 
which can occur with other oversampling techniques.

However, SMOTE also has some limitations and potential drawbacks. 
First, SMOTE may generate noisy synthetic samples if the minority class is highly overlapping with the majority class. 
Second, SMOTE may also introduce bias if the minority class is not well-represented by the existing samples. 
Finally, SMOTE may increase the risk of overfitting if the synthetic samples are too similar to the existing samples.
'''

from imblearn.over_sampling import SMOTE
import pandas as pd

# load imbalanced dataset
df = pd.read_csv('imbalanced_dataset.csv')

# split dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# create SMOTE object
smote = SMOTE(sampling_strategy='minority', random_state=42)
#smote = SMOTE(k_neighbors = 3)

# fit SMOTE on training data and oversample minority class
X_resampled, y_resampled = smote.fit_resample(X, y)

#######################################################################################################################
###################################### Random Over-Sampling Using Imblearn ############################################
#######################################################################################################################

'''
Random Over-Sampling is a data augmentation technique used to address the problem of imbalanced classes in machine learning. 
It involves randomly selecting samples from the minority class and duplicating them until the number of samples in the 
minority class is equal to that in the majority class.
'''

from imblearn.over_sampling import RandomOverSampler
import pandas as pd

# load imbalanced dataset
df = pd.read_csv('imbalanced_dataset.csv')

# split dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# create Random Over-Sampler object
ros = RandomOverSampler(random_state=42)

# fit Random Over-Sampler on training data and oversample minority class
X_resampled, y_resampled = ros.fit_resample(X, y)

#######################################################################################################################
###################################### Random Under-Sampling Using Imblearn ############################################
#######################################################################################################################

'''
Random Under-Sampling is a data augmentation technique used to address the problem of imbalanced classes in machine learning.
 It involves randomly selecting samples from the majority class and removing them until the number of samples in the majority
 class is equal to that in the minority class.
'''

from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# load imbalanced dataset
df = pd.read_csv('imbalanced_dataset.csv')

# split dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# create Random Under-Sampler object
rus = RandomUnderSampler(random_state=42)

# fit Random Under-Sampler on training data and undersample majority class
X_resampled, y_resampled = rus.fit_resample(X, y)

#######################################################################################################################
###################################### Random Over-Sampling ###########################################################
#######################################################################################################################

import pandas as pd
import numpy as np

# load imbalanced dataset
df = pd.read_csv('imbalanced_dataset.csv')

# split dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# count the number of samples in each class
class_counts = np.unique(y, return_counts=True)[1]

# determine the majority and minority class
majority_class = np.argmax(class_counts)
minority_class = np.argmin(class_counts)

# get indices of the minority class
minority_indices = np.where(y == minority_class)[0]

# randomly sample from the minority class with replacement
num_minority_samples = class_counts[majority_class] - class_counts[minority_class]
minority_samples = np.random.choice(minority_indices, size=num_minority_samples, replace=True)

# combine majority class with oversampled minority class
majority_samples = np.where(y == majority_class)[0]
oversampled_indices = np.concatenate((majority_samples, minority_samples))

# create new oversampled dataset
X_resampled = X.iloc[oversampled_indices]
y_resampled = y.iloc[oversampled_indices]

#######################################################################################################################
###################################### Random Under-Sampling ###########################################################
#######################################################################################################################

import pandas as pd
import numpy as np

# load imbalanced dataset
df = pd.read_csv('imbalanced_dataset.csv')

# split dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# count the number of samples in each class
class_counts = np.unique(y, return_counts=True)[1]

# determine the majority and minority class
majority_class = np.argmax(class_counts)
minority_class = np.argmin(class_counts)

# get indices of the majority class
majority_indices = np.where(y == majority_class)[0]

# randomly sample from the majority class without replacement
num_majority_samples = class_counts[minority_class]
majority_samples = np.random.choice(majority_indices, size=num_majority_samples, replace=False)

# combine majority class with undersampled minority class
minority_samples = np.where(y == minority_class)[0]
undersampled_indices = np.concatenate((minority_samples, majority_samples))

# create new undersampled dataset
X_resampled = X.iloc[undersampled_indices]
y_resampled = y.iloc[undersampled_indices]
