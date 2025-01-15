import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataSet = pd.read_csv('Social_Network_Ads.csv')
x = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# print(x_train)
# print(x_test)

classifier = LogisticRegression(random_state = 0)
