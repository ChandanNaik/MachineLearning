import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
class DataA:
    def __init__(self):
        f = lambda x, y : np.random.randn(x, y)
        self.train_x = f(1000, 20)
        self.train_y = f(1000, 1)
        self.test_x = f(500, 20)
        self.test_y = f(500, 1)
        
class DataB:
    def __init__(self):
        # Data from: https://archive.ics.uci.edu/ml/datasets/Cloud
        data = np.fromfile("data/cloud.data", sep = " ").reshape((1024, 10))
        y = data[:, 6]
        X = np.delete(data, 6, axis = 1)
        
        self.train_x = X[:800]
        self.train_y = y[:800]
        
        self.test_x = X[800:]
        self.test_y = y[800:]
        
class DataC:
    def __init__(self):
        # Data from: http://archive.ics.uci.edu/ml/datasets/Forest+Fires
        data = pd.read_csv("data/forestfires.csv")
        data = data.sample(frac = 1).reset_index(drop = True).drop(columns = ["month", "day"])
        data["area"] = np.log(data["area"] + 1)
        X = data.drop(columns = "area").values
        y = data["area"].values
        
        self.train_x = X[:400]
        self.train_y = y[:400]
        
        self.test_x = X[400:]
        self.test_y = y[400:]

data_a = DataA()
data_b = DataB()
data_c = DataC()

# Lasso 

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

datasets = [data_a, data_b, data_c]
datasetsName = ["A", "B", "C"]
regularizationParamArray = [0.01,0.05,0.1,0.2,0.3]
minMeanSquareError = []
minMeanSquareErrorIndex = []
minimumZeros = []
minimumZerosIndex = []
for dataset, datasetName in zip(datasets, datasetsName):
    print("For Dataset : ",datasetName)
    nonZeroCoeffecients = []
    meanSquareErrorData = []
    
    for regularization in regularizationParamArray:
        lassoClassifier = linear_model.Lasso(alpha=regularization)

        lassoClassifier.fit(dataset.train_x, dataset.train_y)
        nonZeroCoeffecients.append(len(np.where(lassoClassifier.coef_==0)[0]))

        y_predicted = lassoClassifier.predict(dataset.test_x)
        mse = mean_squared_error(dataset.test_y, y_predicted)
        meanSquareErrorData.append(mse)
        
    for i,j,k in zip(regularizationParamArray,nonZeroCoeffecients, meanSquareErrorData):
        print("For\t 1. Regularization Parameter: " + str(i) + "\n \t 2. Number of Zeros: " + str(j) + "\n \t 3. Mean Squared Error (MSE) is: " + str(k))
        print("\n")

    minZeros = min(nonZeroCoeffecients)
    minZerosIndex = min(range(len(nonZeroCoeffecients)), key = nonZeroCoeffecients.__getitem__)
    minimumZeros.append(minZeros)
    minimumZerosIndex.append(minZerosIndex)
    
    mse_min = min(meanSquareErrorData)
    minIndexMeanSquareError = min(range(len(meanSquareErrorData)), key = meanSquareErrorData.__getitem__)
    minMeanSquareError.append(mse_min)
    minMeanSquareErrorIndex.append(minIndexMeanSquareError)
    
    print("\n")
    plt.title("Regularization Parameter vs Number of Zeros")
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Non-Zero Count")
    plt.plot(regularizationParamArray, nonZeroCoeffecients, color='green')
    plt.show()

    print("\n")
    plt.title("Regularization Parameter vs Mean Squared Error")
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Mean Squared Error")
    plt.plot(regularizationParamArray, meanSquareErrorData, color='red')
    plt.show()


for i,j in enumerate(datasetsName):
    print("For Dataset : ",j)
    print("Minimum Number of Zeros is: " + str(minimumZeros[i]) + " and is achieved for Regularization Parameter (λ): " + str(regularizationParamArray[minimumZerosIndex[i]]))
    print("Minimum Error is: " + str(minMeanSquareError[i]) + " and is achieved for Regularization Parameter (λ): " + str(regularizationParamArray[minMeanSquareErrorIndex[i]]))
    print("\n")    

# Ridge Regression

from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

datasets = [data_a, data_b, data_c]
datasetsName = ["A", "B", "C"]
regularizationParamArray = [0.01,0.05,0.1,0.2,0.3]
minMeanSquareError = []
minMeanSquareErrorIndex = []
minimumZeros = []
minimumZerosIndex = []

for dataset, datasetName in zip(datasets, datasetsName):
    print("For Dataset : ",datasetName)
    nonZeroCoeffecients = []
    meanSquareErrorData = []

    for regularization in regularizationParamArray:
        ridgeClassifier = Ridge(alpha=regularization)

        ridgeClassifier.fit(dataset.train_x, dataset.train_y)
        nonZeroCoeffecients.append(len(np.where(ridgeClassifier.coef_==0)[0]))

        y_predicted = ridgeClassifier.predict(dataset.test_x)
        mse = mean_squared_error(dataset.test_y, y_predicted)
        meanSquareErrorData.append(mse)

    for i,j,k in zip(regularizationParamArray,nonZeroCoeffecients, meanSquareErrorData):
        print("For\t 1. Regularization Parameter: " + str(i) + "\n \t 2. Number of Zeros: " + str(j) + "\n \t 3. Mean Squared Error (MSE) is: " + str(k))
        print("\n")

    zeros_min = min(nonZeroCoeffecients)
    index_zeros_min = min(range(len(nonZeroCoeffecients)), key = nonZeroCoeffecients.__getitem__)
    minimumZeros.append(zeros_min)
    minimumZerosIndex.append(index_zeros_min)
    
    mse_min = min(meanSquareErrorData)
    index_mse_min = min(range(len(meanSquareErrorData)), key = meanSquareErrorData.__getitem__)
    minMeanSquareError.append(mse_min)
    minMeanSquareErrorIndex.append(index_mse_min)
    
    print("\n")
    plt.title("Regularization Parameter vs Number of Zeros")
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Non-Zero Count")
    plt.plot(regularizationParamArray, nonZeroCoeffecients, color='green')
    plt.show()

    print("\n")
    plt.title("Regularization Parameter vs Mean Squared Error")
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Mean Squared Error")
    plt.plot(regularizationParamArray, meanSquareErrorData, color='red')
    plt.show()


for i,j in enumerate(datasetsName):
    print("For Dataset : ",j)
    print("Minimum Number of Zeros is: " + str(minimumZeros[i]) + " and is achieved for Regularization Parameter (λ): " + str(regularizationParamArray[minimumZerosIndex[i]]))
    print("Minimum Error is: " + str(minMeanSquareError[i]) + " and is achieved for Regularization Parameter (λ): " + str(regularizationParamArray[minMeanSquareErrorIndex[i]]))
    print("\n")