# Perceptron
import unittest

class IrisM:

    def __init__(self):
        from sklearn import datasets
        
        iris = datasets.load_iris()
        
        # only taking first two features
        X = iris.data[:, :2]
        y = iris.target[:]
        
        # only considering whether it is setosa or not
        y[iris.target != 0] = -1
        y[iris.target == 0] = 1
        
        mask = np.random.choice(a = [False, True], size = 150, p = (0.66, 1 - 0.66))
        
        self.train_x, self.train_y = X[mask], y[mask]
        self.test_x, self.test_y = X[~mask], y[~mask]
        
iris = IrisM()
data_train_x = iris.train_x
data_train_y = iris.train_y
print("Shape of train data X", data_train_x.shape)
print("Shape of train data Y", data_train_y.shape)
data_test_x = iris.test_x
data_test_y = iris.test_y
print("Shape of test data X", data_test_x.shape)
print("Shape of test data Y", data_test_y.shape)


class Perceptron:
    def __init__(self, X, y):
        self._X = X
        self._y = y
        self._theta = X[0]  #keeping initial theta value as zero
        self._theta, self._iter = self.train(X, y)
        
    def train(self, X, y):
        tau = 0
        l_rate = 1
        theta = self._theta
        count = 0
        epoch = 0 
        while True:
            epoch += 1 #iterations count
            print(epoch)
            updates = 0
            for i in range(len(X)):
                dotProduct = np.dot(X[i], theta.T) # computing activation
                if dotProduct * y[i] < 0:  #Checking for mistakes
                    updates += 1
                    count += 1
                    theta += X[i] * y[i] * l_rate #updating weights and bias
            if updates == 0:
                break
        
        return [theta, count] 
    
    def predict(self, X):
        predict = np.dot(X, self._theta.T)
        if predict <= 0:
            predict_y = -1
        else:
            predict_y = 1
            
        return predict_y    
        
    def margin(self):

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2], [4, 5], [2, 1], [5, 4]])
        self.y = np.array([+1, +1, -1, -1])
        self.perceptron = Perceptron(self.x, self.y)
        self.queries = np.array([[1, 5], [0, 3], [6, 4], [2, 2]])

    def test0(self):
        """
        Test Perceptron
        """
        self.assertEqual(self.perceptron.predict(self.queries[0]),  1)
        self.assertEqual(self.perceptron.predict(self.queries[1]),  1)
        self.assertEqual(self.perceptron.predict(self.queries[2]), -1)
        self.assertEqual(self.perceptron.predict(self.queries[3]), -1)
        
tests = TestPerceptron()
tests_to_run = unittest.TestLoader().loadTestsFromModule(tests)
unittest.TextTestRunner().run(tests_to_run)
        
    