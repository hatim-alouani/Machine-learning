import numpy as np

print("moindres caree (methode)")
# simple linear regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 14, 18, 22, 26])

# math formulat :
# y = a * x + b
# Calculate means
# a = ∑(x - x_mean) * (y - y_mean)
# b = y_mean - (a * x_mean)

n = len(y)
x_mean = np.mean(x)
y_mean = np.mean(y)

a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - (a * x_mean)

print("simple")
print(f"a  : {a}")
print(f"b  : {b}")
print(f"result  : {a * 5 + b}")

# multiple linear regression

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])

# math formulat : 
# y = a * x + a2 * x2 + b
# Calculate means
# a = ∑(x - x_mean) * (y - y_mean)
# b = y_mean - (a1 * x1_mean) (a2 * x2_mean)

n = len(y)
x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
y_mean = np.mean(y)

a1 = np.sum((x1 - x1_mean) * (y - y_mean)) / np.sum((x1 - x1_mean) ** 2)
a2 = np.sum((x2 - x2_mean) * (y - y_mean))/ np.sum((x2 - x2_mean) ** 2)
b = y_mean - (a1 * x1_mean) - (a2 * x2_mean)

print("multiple")
print(f"a1 : {a1}")
print(f"a2 : {a2}")
print(f"b  : {b}")
print(f"result  : {a1 * 5 + a2 * 6 + b}")