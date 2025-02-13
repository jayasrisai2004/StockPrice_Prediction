#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Load data and preprocess
data = pd.read_csv(r'C:\Users\HI\Downloads\stockprice_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.ffill(inplace=True) 

# Set features and target
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
y = data['Close']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Support Vector Regressor': SVR(kernel='rbf'),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# Dictionary to store RMSE values
model_rmse = {}

# Evaluate each model
for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_rmse[model_name] = rmse

# Output RMSE for each model
print("RMSE for each model:")
for model, rmse in model_rmse.items():
    print(f"{model}: {rmse}")


# In[ ]:




