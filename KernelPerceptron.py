from scipy.io import loadmat
class Data:
    def __init__(self):
        ff = lambda x,y : loadmat(x)[y]
        
        self.X_train = ff("data/iris_3/train_data.mat", "train_data")
        self.y_train = ff("data/iris_3/train_labels.mat", "train_labels").flatten()
        
        self.X_test = ff("data/iris_3/test_data.mat", "test_data")
        self.y_test = ff("data/iris_3/test_labels.mat", "test_labels").flatten()



import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    dotProduct = np.dot(x1, x2)
    return dotProduct
    
def polynomial_kernel(x, y, p = 3):
    kernel = (1 + np.dot(x, y)) ** p
    return kernel
    
def gaussian_kernel(x, y, sigma = 0.5):
    kernel = np.exp(-linalg.norm(x-y)**2/(2*(sigma**2)))
    return kernel        


class KernelPerceptron:
    def __init__(self, kernel = linear_kernel, Niter = 1):
        
        self.kernel = kernel
        self.Niter = Niter
        self.support_vector_x = None
        self.support_vector_y = None
        
    def fit(self, X, y):
        #TO DO
        samples, features = X.shape
        self.tempVector = np.zeros(samples, dtype=np.float64)

        # init and build gram matrix
        gramMatrix = np.zeros((samples, samples))
        for i in range(samples):
            for j in range(samples):
                gramMatrix[i,j] = self.kernel(X[i], X[j])
        
        #Fill tempVector
        i = 0 
        while i in range(self.Niter):  
            for j in range(samples):
                if np.sign(np.sum(gramMatrix[:,j] * self.tempVector * y)) != y[j]:
                    self.tempVector[j] += 1.0
            i += 1

        # Get the support vectors
        vector = self.tempVector > 1e-5
        index = np.arange(len(self.tempVector))[vector]
        self.tempVector = self.tempVector[vector]
        self.support_vector_x = X[vector]
        self.support_vector_y = y[vector]

    def predict(self, X):
        #TO DO
        twoDimensionArray = np.atleast_2d(X)
        projection = np.zeros(len(twoDimensionArray))
        for i in range(len(twoDimensionArray)):
            prediction = 0
            for j, support_vector_y, support_vector_x in zip(self.tempVector, self.support_vector_y, self.support_vector_x):
                prediction += j * support_vector_y * self.kernel(X[i], support_vector_x)
            projection[i] = prediction
        
        return np.sign(projection)


