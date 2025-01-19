import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('salary_data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)
#training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, x_pred, color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#visualization
plt.show()
#test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, x_pred, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#visualization
plt.show()
#predicting salary for 15 years of experience
print(regressor.predict([[15]]))