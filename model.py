# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import warnings
import sklearn

data = pd.read_csv('tokyo.csv')

data.fillna(0, inplace=True)

data.fillna(data.mean(), inplace=True)

# Creating DV and IV sets
X = data.drop('price', axis=1)
y = data['price']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 1234)

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[45, 14, 2, 4, 1, 1, 3, 3, 5474, 3, 0, 1]])) 