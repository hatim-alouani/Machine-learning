import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)#just to display the numbers upto 2 decimal places
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#concatenate the predicted and actual values of y 
#reshape is used to convert the 1D array to 2D array
#1 is the axis along which the arrays will be joined


plt.scatter(range(1,len(y_test)+1),y_test,color='red')
plt.plot(range(1,len(y_test)+1),y_pred,color='blue')
plt.title('Profit vs Test data')
plt.xlabel('Test data')
plt.ylabel('Profit')
plt.show()
