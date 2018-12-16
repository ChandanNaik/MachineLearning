import matplotlib.pylab as plt
%matplotlib inline
import pickle, gzip       
import numpy as np

class Numbers:
    """
    Class to store MNIST data for images of 9 and 8 only
    """ 
    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if you'd like
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
 
        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8
 
        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8

data2 = Numbers('data/mnist.pklz')

def view_digit(example, label=None):
    if label is not None: print("true label: {:d}".format(label))
    plt.imshow(example.reshape(28,28), cmap='gray');
#view_digit(data2.train_x[0],data2.train_y[0])
view_digit(data2.train_x[1],data2.train_y[1])

from collections import defaultdict
class LogReg:
    
    def __init__(self, num_features, eta):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: Learning rate (the default is a constant value)
        """
        
        self.w = np.zeros(num_features)
        self.eta = eta
        
        
    def sgd_update(self, x_i, y):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """
 
        # TODO: Finish this function to do a single stochastic gradient descent update
        # and return the updated weight vector
        
        probability = self.sigmoid(self.w.dot(x_i))
        gradient = np.dot(x_i.T, (probability - y)) / y.size
        self.w -=self.eta * gradient
        return self.w
    
    def sigmoid(self, score, threshold = 20.0):
        """
        Prevent overflow of exp by capping activation at 20.
        :param score: A real valued number to convert into a number between 0 and 1
        """
        # TODO: Finish this function to return the output of applying the sigmoid
        # function to the input score (Please do not use external libraries)
        
        if abs(score) > threshold:
            score = threshold * np.sign(score)
        
        return (1 / (1 + np.exp(-score)))         
    
    def progress(self, examples_x, examples_y):
        """
        Given a set of examples, computes the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """
 
        logprob = 0.0
        num_right = 0
        for x_i, y in zip(examples_x, examples_y):
            p = self.sigmoid(self.w.dot(x_i))
            if y == 1:
                logprob += np.log(p)
            else:
                logprob += np.log(1.0 - p)
 
            # Get accuracy
            if abs(y - p) < 0.5:
                num_right += 1
 
        return logprob, float(num_right) / float(len(examples_y))

# Loop over training data and perform updates
# Sample code:

from sklearn.utils import shuffle

learningRates = [1e-3, 1e-2, 1e-1, 1]
epochs = [5, 10, 15]

for l in learningRates:
    for e in epochs:
        vLoss = []
        vAccuracy = []
        tLoss = []
        tAccuracy = []

        logReg = LogReg(data2.train_x.shape[1], 1e-3)
        i = 0

        for epoch in range(e):
            X_train, Y_train = shuffle(data2.train_x, data2.train_y)
            for x, y in zip(X_train, Y_train):

                i += 1
                p = logReg.sigmoid(logReg.w.dot(x))
                logReg.sgd_update(x, y)

                if i%100 == 0 :
                    trainLoss, trainAccuracy = logReg.progress(data2.train_x, data2.train_y)
                    validLoss, validAccuracy = logReg.progress(data2.valid_x, data2.valid_y)
                    tLoss.append(trainLoss)
                    tAccuracy.append(trainAccuracy)
                    vLoss.append(validLoss)
                    vAccuracy.append(validAccuracy)
                    print("****************************")
                    print("Train Loss : " + str(np.round(trainLoss, 2)))
                    print("Train Accuracy : " + str(np.round(trainAccuracy * 100, 3)) + "%")
                    print("Valid Loss : " + str(np.round(validLoss, 2)))
                    print("Valid Accuracy : " + str(np.round(validAccuracy * 100, 3)) + "%")
                
    plt.title('Accuracy VS Epochs')
    plt.plot(tAccuracy, 'k')
    plt.plot(vAccuracy, 'r')
    plt.legend(['Train Data', 'Test Data'], loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    
    plt.title('Loss VS Epochs')
    plt.plot(tLoss, 'k')
    plt.plot(vLoss, 'r')
    plt.legend(['Train Data', 'Test Data'], loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()   