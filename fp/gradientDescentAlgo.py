import numpy as np

print("gradient descent (optimization algorithm)")
# simple linear regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 14, 18, 22, 26])

# math formulat
# y = a * x + b
# ∂J/∂a = −2/n * ∑x(y − y_pred)
# ∂J/∂b = −2/n * ∑(y − y_pred)
# a = a - learning_rate * ∂J/∂a
# b = b - learning_rate * ∂J/∂b

a = 0
b = 0
n = len(y)

def gradient_descent(x, y, a, b, learning_rate, iterations):
    for _ in range(iterations):
        y_pred = a * x + b
        da = -(2/n) * sum(x * (y - y_pred))
        db = -(2/n) * sum(y - y_pred)
        a = a - learning_rate * da
        b = b - learning_rate * db
    
    return a, b

learning_rate = 0.01
iterations = 1000

a, b = gradient_descent(x, y, a, b, learning_rate, iterations)

print("simple")
print(f"a  : {a}")
print(f"b  : {b}")
print(f"result  : {(a * 5)+ b}")

# multiple linear regression
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])

# math formulat
# y = a1 * x1 + a2 * x2 + b
# ∂J/∂a1 = −2/n * ∑x1(y − y_pred)
# ∂J/∂a2 = −2/n * ∑x2(y − y_pred)
# ∂J/∂b = −2/n * ∑(y − y_pred)
# a1 = a1 - learning_rate * ∂J/∂a1
# a2 = a2 - learning_rate * ∂J/∂a2
# b = b - learning_rate * ∂J/∂b

a1 = 0
a2 = 0
b = 0
n = len(y)
learning_rate = 0.01
iterations = 1000

for _ in range(iterations):
    y_pred = a1 * x1 + a2 * x2 + b
    da1 = -(2/n) * sum(x1 * (y - y_pred))
    da2 = -(2/n) * sum(x2 * (y - y_pred))
    db = -(2/n) * sum(y - y_pred)
    a1 = a1 - learning_rate * da1
    a2 = a2 - learning_rate * da2
    b = b - learning_rate * db


print("multiple")
print(f"a1 : {a1}")
print(f"a2 : {a2}")
print(f"b  : {b}")
print(f"result  : {a1 * 5 + a2 * 6 + b}")