import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough') #One-hot encoding transforms categorical data into a binary matrix where each category gets its own column
x = np.array(ct.fit_transform(x)) #remainder='passthrough': This tells the ColumnTransformer to leave the remaining columns unchanged (without any transformation).
le = LabelEncoder() #This is useful for categorical target variables, where the labels are strings, and you want to convert them into integers
y = le.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])