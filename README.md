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
To read csv file

![EXP2-1](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/8b1b1bf7-e639-4ceb-913e-b3923ae057f2)


To Read Head and Tail Files

![EXP2-2](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/205a4ed2-5813-4570-95c3-c7dbe7279511)



Compare Dataset

![EXP2-31](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/cc60f8f8-a524-4d0d-bb6a-60d99e2a41c7)



Predicted Value

![EXP2-3](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/921c4e15-3be1-4187-aded-c63aadaa8a26)



Graph For Training Set

![EXP2-4](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/1c9e8b15-9199-4037-b22f-f026defe8889)




Graph For Testing Set

![EXP2-5](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/f9ed7cc5-e28a-4376-aac4-efcd9cdc92f9)



Error

![EXP2-6](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/80090da9-868b-4854-b970-840396a4be77)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
