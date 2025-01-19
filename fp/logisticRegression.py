import math
import pandas as pd
import numpy as np

#math formulat 
# sigmoid(z) = 1 / (1 + e^-z) 
# z = ∑(f.w)
# gradient = (1/m) * ∑(y_pred - y) * x
# weights = ∑weights - learning_rate * gradient

#python code
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_probability(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        y_pred = predict_probability(X, weights)
        gradient = (1 / m) * np.dot(X.T, (y_pred - y))
        weights = weights - learning_rate * gradient
    return weights


data = {
    'Maths': [80, 60, 90, 65, 75],
    'Physics': [85, 20, 88, 30, 80],
    'Admission': [1, 0, 1, 0, 1]
}

dataFrame = pd.DataFrame(data)
X = dataFrame.iloc[:, :-1].values
y = dataFrame.iloc[:, -1].values

weights = np.zeros(len(X[0]))

learning_rate = 0.02
iterations = 1000
weights = gradient_descent(X, y, weights, learning_rate, iterations)
newStudent = [75, 60]
probabilite = predict_probability(newStudent, weights)
admit = 1 if probabilite >= 0.5 else 0

print("logistic regression")
if (admit == 1):
    print("admit : oui")
else:
    print("admit : non")