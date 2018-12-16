from sklearn.base import BaseEstimator, ClassifierMixin

class TreeNode:
    def __init__(self, X_train, y_train, height):
        self.left = None
        self.right = None
        self.isLeaf = False
        self.label = None
        self.split_vector = None
        self.X = X_train
        self.Y = y_train
        self.height = height

    def getLabel(self):
        if not self.isLeaf:
            raise Exception("Should not call getLabel on a non-leaf node")
        return self.label
    
class RandomDecisionTree(BaseEstimator, ClassifierMixin):
            
    def __init__(self, candidate_splits = 100, depth = 10):
        """
        Args:
            candidate_splits (int) : number of random decision splits to test
            depth (int) : maximum depth of the random decision tree
        """
        self.candidate_splits = candidate_splits
        self.depth = depth
        self.root = None
    
    def fit(self, X_train, y_train):
        """
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data
            
        """
        self.root = self.build_tree(X_train[:], y_train[:], 0)
        return self
        
        
    def build_tree(self, X_train, y_train, height):
        """
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data
            
        """
        
        node = TreeNode(X_train, y_train, height)
        node.split_vector = self.find_best_split(node.X, node.Y)
        
        if node.split_vector is not None and node.height != self.depth:
            dotProduct = np.dot(node.X, node.split_vector) #l
            X_Left = []
            Y_Left = []
            X_Right = []
            Y_Right = []
            for i in range(len(dotProduct)):
                if dotProduct[i] < 0:
                    X_Left.append(node.X[i])
                    Y_Left.append(node.Y[i])
                else:
                    X_Right.append(node.X[i])
                    Y_Right.append(node.Y[i])
                    
            if len(X_Left) and len(X_Right):
                node.left = self.build_tree(np.array(X_Left), np.array(Y_Left), height + 1)
                node.right = self.build_tree(np.array(X_Right), np.array(Y_Right), height + 1)
            elif len(X_Left) == 0 and len(X_Right):
                node.right = self.build_tree(np.array(X_Right), np.array(Y_Right), height + 1)
            elif len(X_Right) == 0 and len(X_Left):
                node.left = self.build_tree(np.array(X_Left), np.array(Y_Left), height + 1)
        
        else:
            node.isLeaf = True
            node.label = self.majority(node.Y)
            
        return node
    
    
    def find_best_split(self, X_train, y_train):
        """
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data
            
        """        
        totalSplits = self.candidate_splits
        gain_max = 0
        vector = []
        for i in range(totalSplits):
            split_vector = np.random.standard_normal(X_train.shape[1])
            dotProduct = np.dot(X_train, split_vector)
            X_Left = []
            Y_Left = []
            X_Right = []
            Y_Right = []
            for i in range(len(dotProduct)):
                if dotProduct[i] < 0:
                    X_Left.append(X_train[i])
                    Y_Left.append(y_train[i])
                else:
                    X_Right.append(X_train[i])
                    Y_Right.append(y_train[i])
            
            if len(X_Left) and len(X_Right):
                gain = self.gini_index(y_train) - (len(X_Left)/len(X_train))*self.gini_index(np.array(Y_Left)) - (len(X_Right)/len(X_train))*self.gini_index(np.array(Y_Right))
            elif len(X_Left) == 0 and len(X_Right):
                gain = self.gini_index(y_train) - (len(X_Right)/len(X_train))*self.gini_index(np.array(Y_Right))
            elif len(X_Right) == 0 and len(X_Left):
                gain = self.gini_index(y_train) - (len(X_Left)/len(X_train))*self.gini_index(np.array(Y_Left))
            
            if gain > gain_max:
                gain_max = gain
                vector = split_vector
        
        if gain > 0:
            return vector
        else:
            return None
            
        
    def gini_index(self, y):
        """
        Args:
            y (ndarray): [n_samples] ndarray of data
        """
        """Calculate the Gini coefficient of a numpy array."""
        
        # https://github.com/oliviaguest/gini 
        # Gini require non-zero positive (ascending-order) sorted values within a one dimension vector.
        
        yArray = y.flatten() #removes multi dimension
        if np.amin(yArray) < 0:
            yArray -= np.amin(y) #no negatives
        yArray = yArray + 0.0000001 #no zero values in the array
        yArray = np.sort(yArray) 
        index = np.arange(1,y.shape[0]+1) 
        n = y.shape[0]
        return ((np.sum((2 * index - n  - 1) * yArray)) / (n * np.sum(yArray))) #gini value
    
    def majority(self, y):
        """
        Return the major class in ndarray y
        """
        distinctLabel, labelCount = np.unique(y, return_counts = True)
        totalCounts = np.where(labelCount==labelCount.max())
        majority = distinctLabel[totalCounts]
        return majority[0]
    
    def predict(self, X):
        """
        BaggingClassifier prediction for data points in X
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns:
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """
        yhat = list()
        towardsY = self.fit(data.X_train,data.y_train)
        for i in range(0,len(X)):
            ithLabel = self.makePrediction(X[i],towardsY.root)
            yhat.append(ithLabel)
        return np.array(yhat)
    
    def makePrediction(self,i,root):
        while(root.isLeaf == False):
            if np.dot(i, root.split_vector)>0:
                root = root.right
            else:
                root = root.left
        return root.label


# RandomDecisionTree for Handwritten Digit Recognition

from sklearn.metrics import accuracy_score

decisionTree = RandomDecisionTree()
decisionTree.fit(data.X_train, data.y_train)
yPredict = myClassifier.predict(data.X_valid)
print(accuracy_score(data.y_valid, yPredict))


# RandomForest for Handwritten Digit Recognition

from sklearn.metrics import accuracy_score

for i in [2,6,10,15]:
    for j in [0.2,0.4,0.6,0.8,1.0]:
        for k in [2,4,6,8,10]:
            for l in [50,100,150]:
                for train_index, val_index in skf.split(data.X_train): 
                    X_train, X_test = data.X_train[train_index], data.X_train[val_index] 
                    y_train, y_test = data.y_train[train_index], data.y_train[val_index]
                    randomForest = RandomForest(j, i, k, l)
                    randomForest.fit(data.X_train, data.y_train)
                    predict = randomForest.predict(data.X_valid)
                    print("Accuracy with :- Ratio = {}, Samples = {}, Depth = {} and Candidate = {} is {}".format(j ,i ,k ,l ,accuracy_score(data.y_valid, predict)))

"""
From the results it can be seen that the best accuracy can be arrived with the following values
    * Ratio = 1.0, N = 15, Depth = 10 ,Candidates = 150 
"""

print("\n Accuracy using optimal paramaters \n")
randomForest = RandomForest(1.0, 15, 10, 150)
randomForest.fit(data.X_train, data.y_train)
predict = randomForest.predict(data.X_valid)
print("Accuracy : ", accuracy_score(data.y_valid, predict))

