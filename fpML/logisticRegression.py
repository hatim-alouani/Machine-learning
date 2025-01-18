import math
import pandas as pd
import numpy as np

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
    'Physics': [85, 20, 88, 30, 80],
    'Admission': [1, 0, 1, 0, 1]
}

dataFrame = pd.DataFrame(data)
X = dataFrame.iloc[:, :-1].values
y = dataFrame.iloc[:, -1].values

weights = np.zeros(len(X[0]))

learning_rate = 0.02
iterations = 1000
print(f"weights{weights}")
weights = gradient_descent(X, y, weights, learning_rate, iterations)
print(f"weights{weights}")
newStudent = [60, 20]
probabilite = predict_probability(newStudent, weights)
admit = 1 if probabilite >= 0.5 else 0
if (admit == 1):
    print("admit : oui")
else:
    print("admit : non")