# Bagging

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

class BaggingClassifier:
    def __init__(self, ratio = 0.20, N = 20, base=DecisionTreeClassifier(max_depth=4)):
        """
        Create a new BaggingClassifier
        
        Args:
            base (BaseEstimator, optional): Sklearn implementation of decision tree
            ratio: ratio of number of data points in subsampled data to the actual training data
            N: number of base estimator in the ensemble
        
        Attributes:
            base (estimator): Sklearn implementation of decision tree
            N: Number of decision trees
            learners: List of models trained on bootstrapped data sample
        """
        
        assert ratio <= 1.0, "Cannot have ratio greater than one"
        self.base = base
        self.ratio = ratio
        self.N = N
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train Bagging Ensemble Classifier on data
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """
        #TODO: Implement functionality to fit models on the bootstrapped samples
        # cloning sklearn models:
        # from sklearn.base import clone
        # h = clone(self.base)
        
        for i in range(self.N):
            cloneData = clone(self.base)
            X, Y = self.bootstrap(X_train, y_train)
            cloneData.fit(X, Y)
            self.learners.append(cloneData)
        
    def bootstrap(self, X_train, y_train):
        """
        Args:
            n (int): total size of the training data
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """
        ratio = self.ratio
        n_sample = np.random.choice(X_train.shape[0], int(ratio*X_train.shape[0]), replace=True)
        return X_train[n_sample],y_train[n_sample]
    
    def predict(self, X):
        """
        BaggingClassifier prediction for data points in X
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns:
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """
        #TODO: Using the individual classifiers trained predict the final prediction using voting mechanism
        Y_prediction_array = np.empty((X.shape[0], 1))
        for i in range(self.N):
            Y_predict = self.learners[i].predict(X)
            Y_prediction_array = np.concatenate((Y_prediction_array, Y_predict.reshape((Y_predict.shape[0],1))), axis=1)
        
        Y_prediction_array = np.delete(Y_prediction_array, 0, axis = 1)
        
#         assert Y_prediction_array.shape[1] == self.N
#         assert Y_prediction_array.shape[0] == X.shape[0]
        
        Y_FinalPredictions = self.voting(Y_prediction_array)
        return Y_FinalPredictions
    
    def voting(self, y_hats):
        """
        Args:
            y_hats (ndarray): [N] ndarray of data
        Returns:
            y_final : int, final prediction of the 
        """
        #TODO: Implement majority voting scheme and incase of ties return random label
        finalY = []
        for var in y_hats:
            uniqueLabel, labelCount = np.unique(var, return_counts = True)
            totalCounts = np.where(labelCount==labelCount.max())
            mostFrequent = uniqueLabel[totalCounts]
            finalY.append(mostFrequent[0])   
        return np.array(finalY)


# BaggingClassifier for Handwritten Digit Recognition

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
for folds in [5,10,15]:
    N = folds
    for j in [0.3,0.4,0.5,0.7,1.0]:
        ratio = j
        depth = 2 #initial value
        while depth in [2,4,6,8,10]:
            kf = KFold(n_splits=3)
            kf.get_n_splits(data.X_train)
            for train_index, test_index in kf.split(data.X_train):
                X_train, X_test = data.X_train[train_index], data.X_train[test_index]
                y_train, y_test = data.y_train[train_index], data.y_train[test_index]
            base = DecisionTreeClassifier(max_depth=depth)
            classifier = BaggingClassifier(j, N, base)
            classifier.fit(X_train, y_train)
            yPredict = classifier.predict(X_test)
            print("Accuracy with :- Ratio = {}, Samples = {} and Depth = {} is {}".format(j, folds, depth, accuracy_score(y_test, yPredict)))
            depth += 2  
        
""" 
Optimal values:
From the results we can see that following are the best values for greater accuracy:
    * Ratio = 1.0, Samples = 15 and Depth = 10 is 0.9704284852142426 i.e., 97.04 => 97% accuracy.
"""
base = DecisionTreeClassifier(max_depth=10)
classifier = BaggingClassifier(1.0, 15, base)
classifier.fit(data.X_train, data.y_train)
predict = classifier.predict(data.X_test)
print("Accuracy with :- Ratio = 1.0, Samples = 15 and Depth = 10 is {}".format(accuracy_score(data.y_valid, predict)))


