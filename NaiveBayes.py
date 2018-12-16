from scipy.io import loadmat
import numpy as np

class SPECT:
    def __init__(self):
        ff = lambda x, y : loadmat(x)[y]
        
        self.X_train = ff('data/SPECTtrainData.mat','trainData')
        self.y_train = ff('data/SPECTtrainLabels.mat','trainLabels')
        
        self.X_test = ff('data/SPECTtestData.mat', 'testData')
        self.y_test = ff('data/SPECTtestLabels.mat', 'testLabels')
        
        # Label normal : 1 abnormal : 0
data1 = SPECT()


class NaiveBayes:
    def __init__(self, n = 2, prior = 0.5):
        """
        Create a NaiveBayes classifier
        :param n : small integer
        :param prior: prior estimate of the value of pi
        """
        
        self.n = n
        self.prior = prior
        self.normal_model = None
        self.abnormal_model = None

    
    def train(self,xNormal,xAbnormal,pNormal,pAbnormal):
        #Trainer
        for feature in range(xNormal.shape[1]):
            columnNormal = xNormal[:,feature]
            columnAbnormal = xAbnormal[:, feature]
            pNormal[feature] = ((columnNormal == 1).sum() + self.n * self.prior) / (xNormal.shape[0] + self.n)
            pAbnormal[feature] = ((columnAbnormal == 1).sum() + self.n * self.prior) / (xAbnormal.shape[0] + self.n)
            
        return pNormal,pAbnormal
    
    def fit(self, X_train, y_train):
        """
        Generate probabilistic models for normal and abmornal group.
        Use self.normal_model and self.abnormal_model to store 
        models for normal and abnormal groups respectively
        """
        #TODO: Finish this function
        a,b = np.unique(y_train, return_counts = True)
        xNormal = []
        xAbnormal = []
        
        for i,j in zip(X_train,y_train): #mapping similar items
            if j == 0:
                xAbnormal.append(i)
            else:
                xNormal.append(i)
        
        xNormal = np.array(xNormal)
        xAbnormal = np.array(xAbnormal)
        pNormal = np.zeros(xNormal.shape[1])
        pAbnormal = np.zeros(xAbnormal.shape[1])
        
        pN,pAb = NaiveBayes.train(self,xNormal,xAbnormal,pNormal,pAbnormal)
        self.normal_model = pN
        self.abnormal_model = pAb
        
    def predict(self, data):
        """
        Return predicted label for the input example
        :param data: input example
        """
            
        #TODO: Finish this function
        normalProbability = 1
        abnormalProbability = 1

        for i in range(data.size):
            normalProbability = normalProbability * (self.normal_model[i]**data[i]) * (1-self.normal_model[i])**(1-data[i])
            abnormalProbability = abnormalProbability * (self.abnormal_model[i]**data[i]) * (1-self.abnormal_model[i])**(1-data[i])
        
        if normalProbability > abnormalProbability:
            return 1
        else:
            return 0

naiveBayes = NaiveBayes()
naiveBayes.fit(data1.X_train, data1.y_train)

predict = 0
for i in range(data1.X_test.shape[0]):
    y = naiveBayes.predict(data1.X_test[i])
    if y == data1.y_test[i]:
        predict +=1
        
errorRate = (1 - (float(predict)/float(data1.X_test.shape[0])))
print("Error rate is " + str(round(errorRate * 100, 3)) + "%")