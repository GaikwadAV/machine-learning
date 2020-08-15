import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
dataset = pd.read_csv('HeadBrain.csv')
x = dataset.iloc[:,2:-1]
print('x column',x)
y = dataset.iloc[:,3:]
print('y column',y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title(' Head_Size(cm^3) vs Brain_Weight(grams)(Training set)')
plt.xlabel('Head_Size(cm^3)')
plt.ylabel('Brain_Weight(grams)')
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title(' Head_Size(cm^3) vs Brain_Weight(grams)(testing set)')
plt.xlabel('Head_Size(cm^3)')
plt.ylabel('Brain_Weight(grams)')
plt.show()
