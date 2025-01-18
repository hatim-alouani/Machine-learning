import numpy as np
import math
x = np.array([50,60,80,100,120])
y = np.array([150,180,220,280,300])
n = 5

#math formula
#y = a * x + b
#a = n⋅∑(x.y)−∑(x)⋅∑(y) / n⋅∑(x**2)−∑(x)**2
#b = ∑(y)−a⋅∑(x) / n

#python code
a = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x**2) - sum(x)**2)
b = (sum(y) - a * sum(x)) / n

def gradient_descent(x, y, a, b, learning_rate, iterations):
    n = len(y)
    for _ in range(iterations):
        # Predicted values
        y_pred = a * x + b
        
        # Gradients
        #math formula : da = ∂J/∂a = −2/n * ∑x(y − y_pred)
        da = -(2/n) * sum(x * (y - y_pred))
        #math formula : db = ∂J/∂b = −2/n * ∑(y − y_pred)
        db = -(2/n) * sum(y - y_pred)
        
        # Update parameters
        #math formula : a = a - learning_rate * ∂J/∂a
        a = a - learning_rate * da
        #math formula : b = b - learning_rate * ∂J/∂b
        b = b - learning_rate * db
    
    return a, b

x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 4, 6, 8, 10]) # Dependent variable

learning_rate = 0.01
iterations = 1000

print(f"a : {a}")
print(f"b : {b}")
#y = a * x + b
print(f"y : {(a * 90)+ b}")
a, b = gradient_descent(x, y, a, b, learning_rate, iterations)
print(f"a after optimization : {a}")
print(f"b after optimization : {b}")
print(f"y after optimization : {(a * 90)+ b}")