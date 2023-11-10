plt.imshow(X[0], cmap='gray')

X_updated = X.reshape(len(X), -1)
X_updated.shape
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)
