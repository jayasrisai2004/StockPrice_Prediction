#!/usr/bin/env python
# coding: utf-8

# Import Required Libraries

# In[208]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler


# Load Data

# In[210]:


d1 = pd.read_csv('AXISBANK.NS.csv')
d2 = pd.read_csv('ICICIBANK.NS.csv')
d3 = pd.read_csv('HDFCBANK.NS.csv')
d4 = pd.read_csv('INDUSINDBK.NS.csv')
d5 = pd.read_csv('KOTAKBANK.NS.csv')


# Sort and Save Merged Data

# In[212]:


merged_data = pd.concat([d1, d2,d3,d4,d5], axis=0)
data = merged_data.sort_values(by="Date", ascending=True) 
data.to_csv("stockprice_dataset.csv",index=False)
data


# Data Overview

# In[214]:


data.columns.tolist()


# In[215]:


data.head()


# In[216]:


data.tail()


# In[217]:


data.info()


# In[218]:


data.describe()


# In[219]:


data.nunique()


# In[220]:


data['Date'].value_counts()


# In[221]:


data.shape


# In[222]:


data.isnull().sum()


# In[223]:


data.isnull().sum().values.sum()


# In[224]:


data.dropna(inplace=True)
data.isnull().sum().values.sum()


# In[225]:


data.isnull().sum()


# In[226]:


data.shape


# In[227]:


data.dtypes


# In[228]:


if 'Date' in data.columns:
    plt.figure(figsize=(15, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.title("Time Series of Close Price")
    plt.xlabel("Year")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()


# Outlier Detection and Removal

# In[230]:


num_cols = data.select_dtypes(include=[np.number]).columns


# In[231]:


for col in num_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1  # Interquartile range
        
        # Define lower and upper bounds for acceptable data points
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out rows with values outside of the lower and upper bounds
        cleaned_data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]


# In[232]:


cleaned_data


# In[233]:


cleaned_data.shape


# Visualize Outliers with Boxplots

# In[235]:


num_cols = data.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns

# 1. Boxplots for Outlier Visualization
plt.figure(figsize=(30, 20))
for i, col in enumerate(num_cols, 1):
    # Original data boxplot
    plt.subplot(2, len(num_cols), i)
    sns.boxplot(y=data[col])
    plt.title(f'Original: {col}')
    
    # Cleaned data boxplot
    plt.subplot(2, len(num_cols), i + len(num_cols))
    sns.boxplot(y=cleaned_data[col])
    plt.title(f'Cleaned: {col}')
    
plt.tight_layout()
plt.show() 


# In[236]:


# Calculate the correlation matrix
correlation_matrix = cleaned_data.corr(numeric_only=True)

# Display the entire correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[237]:


# Display correlation of all features with the 'Close' column
close_correlation = correlation_matrix['Close'].sort_values(ascending=False)
print("\nCorrelation with Close:")
print(close_correlation)


# In[238]:


# Select a subset of columns for analysis
subset_data = cleaned_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Create the pair plot
pairplot = sns.pairplot(subset_data, diag_kind='kde', plot_kws={'color': 'purple'}, diag_kws={'color': 'green'})
pairplot.fig.suptitle('Pairplot for Open, High, Low, Close, Adj Close and Volume', y=1.02)

plt.show()


# In[239]:


# Define a custom color palette
custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Blue, Orange, Green, Red, Purple

# 5. Scatter Plots between Volume and Close (or other pairs)
plt.figure(figsize=(12, 6))

# Scatter plot with the first color in the custom palette
scatter = sns.scatterplot(data=subset_data, x="Volume", y="Close", alpha=0.6, s=100, edgecolor='w', linewidth=0.5, color=custom_colors[0])

# Add a regression line in a contrasting color (e.g., orange)
sns.regplot(data=subset_data, x="Volume", y="Close", scatter=False, color=custom_colors[1], line_kws={"linewidth": 2})

# Title and labels with increased font sizes
plt.title("Scatter Plot of Volume vs. Close Price", fontsize=18)
plt.xlabel("Volume", fontsize=14)
plt.ylabel("Close Price", fontsize=14)

# Add grid lines for better visibility
plt.grid(True)

# Set limits (optional, adjust as necessary based on your data)
plt.xlim(subset_data["Volume"].min() - 1, subset_data["Volume"].max() + 1)
plt.ylim(subset_data["Close"].min() - 1, subset_data["Close"].max() + 1)

plt.show()


# In[240]:


cleaned_data


# Feature Engineering

# In[242]:


import pandas as pd

# Create a copy of cleaned_data to avoid SettingWithCopyWarning
cleaned_data = cleaned_data.copy()

# Check if 'Date' column exists in cleaned_data
if 'Date' not in cleaned_data.columns:
    print("The 'Date' column is missing.")
else:
    # Convert 'Date' to datetime format and check for errors
    cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], errors='coerce')
    
    # Check for any rows where 'Date' conversion resulted in NaT (Not a Time, indicating failed conversion)
    if cleaned_data['Date'].isna().any():
        print("Warning: Some dates could not be converted and are set as NaT. Consider handling them.")
        # Optionally, you can drop rows with NaT in 'Date' like so:
        cleaned_data = cleaned_data.dropna(subset=['Date'])

    # Extract useful features from the Date column if Date conversion was successful
    cleaned_data['Day_of_Week'] = cleaned_data['Date'].dt.dayofweek  # Monday=0, Sunday=6
    cleaned_data['Month'] = cleaned_data['Date'].dt.month            # Month as a number (1-12)
    cleaned_data['Quarter'] = cleaned_data['Date'].dt.quarter        # Quarter (1-4)
    cleaned_data['Year'] = cleaned_data['Date'].dt.year              # Year (useful if spanning multiple years)

    # Drop the original Date column
    cleaned_data.drop(columns=['Date'], inplace=True)

# Calculate additional columns for price change, high-low difference, and daily return
cleaned_data['Price_Change'] = cleaned_data['Close'] - cleaned_data['Open']  # Daily price change
cleaned_data['High_Low_Diff'] = cleaned_data['High'] - cleaned_data['Low']   # Daily high-low difference
cleaned_data['Return'] = cleaned_data['Close'].pct_change()                  # Daily return


# In[243]:


cleaned_data


# In[244]:


cleaned_data = cleaned_data.dropna()
print(cleaned_data.shape)


# In[245]:


target = 'Close'  
correlations_with_target = cleaned_data.corr()[target]  # Compute correlation with the target


# Step 4: Select features based on correlation threshold
correlation_threshold = 0.8  
selected_features = correlations_with_target[correlations_with_target.abs() >= correlation_threshold].index.tolist()

# Remove the target variable itself from the selected features
selected_features.remove(target)
print("\nSelected features based on correlation threshold:", selected_features)

# Create a subset of the data with only the selected features and the target
data_selected = cleaned_data[selected_features + [target]]

# Optional: Visualize correlations among selected features
plt.figure(figsize=(10, 8))
sns.heatmap(data_selected.corr(), annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Matrix for Selected Features")
plt.show()

# Step 5: Separate features and target for model training
X = data_selected[selected_features]  # Selected features as input
y = data_selected[target]  # Target variable


print("Selected features shape:", X.shape)
print("Target variable shape:", y.shape)


# In[246]:


# Step 4: Scaling
#numerical_cols = data_selected.select_dtypes(include=['float64', 'int64']).columns
#scaler = MinMaxScaler()

# Use .loc to avoid SettingWithCopyWarning
#data_selected.loc[:, numerical_cols] = scaler.fit_transform(data_selected[numerical_cols])



# In[247]:


# Check scaled data (optional)
#print("Scaled Features:")
#print(data_selected.head())


# In[249]:


# Step 5: Train-Test Split
X = data_selected[selected_features]
y = data_selected[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[250]:


X_train


# In[251]:


y_train


# In[252]:


X_test


# In[253]:


y_test


# In[254]:


# Step 2: Initialize the Random Forest model
random_forest_model = RandomForestRegressor(random_state=42, n_estimators=100)
# Step 3: Train the model
random_forest_model.fit(X_train, y_train)
# Step 4: Make predictions on the test set
y_pred = random_forest_model.predict(X_test)




# In[255]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualization 1: Predicted vs Actual Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Close Prices', color='blue', linestyle='dashed', alpha=0.7)
plt.plot(y_pred, label='Predicted Close Prices', color='orange', alpha=0.7)
plt.title("Actual vs Predicted Close Prices - Random Forest")
plt.xlabel("Observations")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Visualization 2: Residuals
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals, color='purple', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title("Residual Plot - Random Forest")
plt.xlabel("Predicted Close Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Visualization 3: Distribution of Residuals
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, bins=30, color='green', alpha=0.6)
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Visualization 4: Scatter Plot of Predicted vs Actual
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test.values, y=y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Perfect prediction line
plt.title("Predicted vs Actual Close Prices")
plt.xlabel("Actual Close Prices")
plt.ylabel("Predicted Close Prices")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[256]:


# Calculate performance metrics

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100



# In[257]:


# Print the metrics
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R² Score: {r2}")
print(f"MAPE: {mape}%")
print(f"Explained Variance Score: {explained_variance}")


# In[318]:


# Example selected features used during training (modify based on your actual training setup)
selected_features = ['Open', 'High', 'Low', 'Adj Close', 'Year']  # Ensure this matches your training data

# Sample input data for prediction
sample_data = {
    'Open': [1699.150024],
    'High': [1700.0],
    'Low': [1675.0],
    'Adj Close': [1679.699951],
    'Year': [2023]
}

# Create a DataFrame for the sample data
sample_df = pd.DataFrame(sample_data, columns=selected_features)

# If you used scaling during training, make sure to scale the data
# Assuming 'scaler' was fitted on the training data
#sample_scaled = scaler.transform(sample_df)

# Assuming 'random_forest_model' is your trained model
predicted_prices = random_forest_model.predict(sample_df)

# Output the predicted Close prices
print("Predicted Close Prices:", predicted_prices)


# In[ ]:





# In[ ]:





# In[ ]:




