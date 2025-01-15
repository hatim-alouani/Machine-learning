import math
import pandas as pd
import numpy as np

#math formula : sigmoid(z) = 1 / (1 + e^-z)
#z = ∑(f.w)
#f is the feature vector and w is the weight vector of the model 
#prediction = sigmoid(z)
#Sigmoid function is used to convert the output of the linear equation to a probability

#cost function = -1/m * ∑(y * log(prediction) + (1 - y) * log(1 - prediction))
#cost function is used to measure the error of the model

#gradient = ∑((prediction - y) * xi)
#graddient is the derivative of the cost function
#y is the actual value and xi is the feature value

#weights = weights - learning_rate * gradient / m
#weights are updated using the gradient and the learning rate
#m is the number of samples in the dataset
#the learning rate is a hyperparameter that controls how much the weights are updated

#python code
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_probability(features, weights):
    z = sum(f * w for f, w in zip(features, weights))
    return sigmoid(z)

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradients = np.zeros(len(weights))
        for i in range(m):
            xi = X[i]
            yi = y[i]
            prediction = predict_probability(xi, weights)
            for j in range(len(weights)):
                gradients[j] += (prediction - yi) * xi[j]
        for j in range(len(weights)):
            weights[j] -= learning_rate * gradients[j] / m
    return weights

data = {
    'Maths': [80, 60, 90, 65, 75],
    'Physics': [85, 70, 88, 60, 80],
    'Admission': [1, 0, 1, 0, 1]
}

dataFrame = pd.DataFrame(data)
X = dataFrame.iloc[:, :-1].values
y = dataFrame.iloc[:, -1].values

weights = np.zeros(len(X[0]))

learning_rate = 0.01
iterations = 10000

weights = gradient_descent(X, y, weights, learning_rate, iterations)

newStudent = [600, 70]
probabilite = predict_probability(newStudent, weights)
admit = 1 if probabilite >= 0.5 else 0
if (admit == 1):
    print("admit : oui")
else:
    print("admit : non")