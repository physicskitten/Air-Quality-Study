import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
df = pd.read_csv("USB/Air-Quality-Study/AirQuality.csv", sep=";", decimal=",")
print("First few rows of the dataset:")
print(df.head())

# Visualize missing data using a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isna(), yticklabels=False, cmap='crest', cbar=False)
plt.title("Heatmap of Missing Data")
plt.show()

# Remove unnecessary columns (e.g., unnamed index columns)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Dataset after removing unnecessary columns:")
print(df.head())

# Replace -200 values with NaN (missing data)
df.replace(to_replace=-200, value=np.nan, inplace=True)

# Drop rows with missing data
df.dropna(inplace=True)
print("Dataset after dropping rows with missing data:")
print(df.tail())

# Feature Engineering: Extract 'Month' from 'Date' if applicable
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month

# Visualization: Histogram of CO(GT)
plt.figure(figsize=(10, 6))
df['CO(GT)'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of CO(GT)')
plt.xlabel('CO(GT)')
plt.ylabel('Frequency')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Visualization: Average CO(GT) by Month
monthly_avg = df.groupby('Month')['CO(GT)'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette='crest')
plt.title('Average CO(GT) by Month')
plt.xlabel('Month')
plt.ylabel('Average CO(GT)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Additional Data Exploration: Summary statistics
print("Summary statistics of the dataset:")
print(df.describe())

