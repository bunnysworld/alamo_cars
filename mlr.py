import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data=pd.read_csv('taxi.csv')

data_x=data.iloc[:,:-1].values
data_y=data.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3,random_state=0)

reg=LinearRegression()
reg.fit(x_train,y_train)

pickle.dump(reg,open('cars.pkl','wb'))
model=pickle.load(open('cars.pkl','rb'))
print(model.predict([[90,1777772,5000,69]]))