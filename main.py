"""
    Honey Production
As you may have already heard, the honeybees are in a precarious state right now.
You may have seen articles about the decline of the honeybee population for various reasons.
You want to investigate this decline and how the trends of the past predict the future for the honeybees.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# dataset Honey Production in the USA (1998-2012)
# Source https://www.kaggle.com/datasets/jessicali9530/honey-production
df = pd.read_csv("honeyproduction.csv")

# 1. state, numcol ,yieldpercol, totalprod, stocks, priceperlb, prodvalue, year
print(df.head)

# 2 total production of honey per year using .groupby() method provided by pandas to get the mean per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# 3 columnOfInterest = df['columnName'] then reshape it to get it into the right format
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# 4 total product per year
y = prod_per_year['totalprod']

# 5 Using plt.scatter(), plot y vs X as a scatterplot.
plt.scatter(X, y)
# Display the plot
# plt.show()

# 6 Create a linear regression model from scikit-learn
# var = module.constructor
regr = linear_model.LinearRegression()

# 7 Fit the model to the data by using .fit()
# feed X into your regr model by passing it in as a parameter of .fit()
regr.fit(X, y)

# 8 The slope of the line will be the first (and only) element of the regr.coef_ list.
print(regr.coef_[0])
print(regr.intercept_)

# 9 Create a list called y_predict that is the predictions
# of the linear regression model
y_predict = regr.predict(X)

# 10 Plot y_predict vs X as a line, on top of the scatterplot
plt.plot(X, y_predict)
# plt.show()

# 11a production of honey has been in decline,
# according to this linear model. Let’s predict what the year 2050 may look like in terms of honey production.
X_future = np.array(range(2013, 2051))
# 11b You can think of reshape() as rotating this array.
# Rather than one big row of numbers, X_future is now a big column of numbers — there’s one number in each row.
X_future = X_future.reshape(-1, 1)

# 12 a list called future_predict that is the y-values that your regr model would predict for the values of X_future.
future_predict = regr.predict(X_future)

# 13 Plot future_predict vs X_future
plt.plot(X_future, future_predict)
plt.show()
