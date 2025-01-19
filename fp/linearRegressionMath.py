import numpy as np

print("linear regression (method)")
#simple linear regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 14, 18, 22, 26])

#math formula
#y = a * x + b
#a = n⋅∑(x.y)−∑(x)⋅∑(y) / n⋅∑(x**2)−∑(x)**2
#b = ∑(y)−a⋅∑(x) / n

n = len(y)

a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
b = (np.sum(y) - a * np.sum(x)) / n

print("simple linear regression")
print(f"a  : {a}")
print(f"b  : {b}")
print(f"result  : {(a * 5)+ b}")

#multiple linear regression
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])

#math formula
#y = a1 * x1 + a2 * x2 + b
#a1 = n⋅∑(x1.y)−∑(x1)⋅∑(y) / n⋅∑(x1**2)−∑(x1)**2
#a2 = n⋅∑(x2.y)−∑(x2)⋅∑(y) / n⋅∑(x2**2)−∑(x2)**2
#b = (∑(y) − a1⋅∑(x1) - a2⋅∑(x2)) / n

n = len(y)

a1 = (n * np.sum(x1 * y) - np.sum(x1) * np.sum(y)) / (n * np.sum(x1**2) - np.sum(x1)**2)
a2 = (n * np.sum(x2 * y) - np.sum(x2) * np.sum(y)) / (n * np.sum(x2**2) - np.sum(x2)**2)
b = (np.sum(y) - (a1 * np.sum(x1)) - (a2 * np.sum(x2))) / n

print("multiple linear regression")
print(f"a1  : {a1}")
print(f"a2  : {a2}")
print(f"b  : {b}")
print(f"result  : {a1 * 5 + a2 * 6 + b}")
