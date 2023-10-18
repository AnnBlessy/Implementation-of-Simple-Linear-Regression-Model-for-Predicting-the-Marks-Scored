# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm
1.import the needed packages. 
2. Assigning hours to x and scores to y. 
3. Plot the scatter plot. 
4. Use mse,rmse,mae formula to find the values.


## Program:
```
Developed by: ANN BLESSY PHILIPS
RegisterNumber: 212222040008

# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:

df.head()

![EXP2-2(a)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/83e836a5-4565-48a7-9977-a7f085fe48d5)

df.tail()

![EXP2-2(b)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/1e89f794-316e-495d-8cc6-d996bf4aa8fb)




Array value of X

![EXP2-3(a)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/6f0f203a-f964-45a6-8fef-03c94460ab0f)


Array value of Y

![EXP2-3(b)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/b395273b-5b41-401e-b75e-ece609b0ba79)


Values of Y Prediction

![EXP2-3(c)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/e3e7d3bb-9a18-4345-a77c-a82898b9bee6)


Array Values of Y test

![EXP2-3(d)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/e0268633-13d2-49c9-9a4a-b970839acf42)





Graph For Training Set

![EXP2-4](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/1c9e8b15-9199-4037-b22f-f026defe8889)






Graph For Testing Set

![EXP2-5](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/f9ed7cc5-e28a-4376-aac4-efcd9cdc92f9)



Error

![EXP2-6](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/80090da9-868b-4854-b970-840396a4be77)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
