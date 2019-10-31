from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import util


def correlation(x,y):
    linreg = linear_model.LinearRegression()
    linreg.fit(x,y)

    return float(linreg.coef_)



def LinReg(x,y):

  #split data into train & test
  x_train,y_train,x_test,y_test = util.split_data(x,y)

  #fit linear regression model
  linreg = linear_model.LinearRegression()
  linreg.fit(x_train,y_train)

  #make predictions on test data
  y_pred = linreg.predict(x_test)

  #evaluate performance
  print("Mean squared error: %.2f" %mean_squared_error(y_test,y_pred))
  print('Variance score: %.2f' % r2_score(y_test, y_pred))

  #plot test data & predictions on test data
  plt.scatter(x_test, y_test,  color='black')
  plt.plot(x_test, y_pred, color='blue', linewidth=3)
  plt.show()



def RANSAC(x,y):

   #split data into train & test
   x_train,y_train,x_test,y_test = util.split_data(x,y)

   #fit linear regression model
   linreg = linear_model.RANSACRegressor()
   linreg.fit(x_train,y_train)

   #make predictions on test data
   y_pred = linreg.predict(x_test)

   #evaluate performance
   print("Mean squared error: %.2f" %mean_squared_error(y_test,y_pred))
   print('Variance score: %.2f' % r2_score(y_test, y_pred))

   #plot test data & predictions on test data
   plt.scatter(x_test, y_test,  color='black')
   plt.plot(x_test, y_pred, color='blue', linewidth=3)
   plt.show()
