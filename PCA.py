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

pca.components_ # Components data2

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
