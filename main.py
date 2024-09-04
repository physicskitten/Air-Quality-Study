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

# Since the data is real valued, we should replace all the null values with mean of each column
for i in col:
    df[i] = df[i].fillna(df[i].mean())

df.isna().sum

# Plot box plots for summary statistics of numerical columns
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.title('Box Plot for Outliers checking')
plt.xticks(rotation=45)
plt.show()

# getting the quartile one and quartile 3 values of each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
# finally calculating the interquartile range IQR
IQR = Q3 - Q1

# if the values fall behind Q1 - (1.5 * IQR) or above Q3 + 1.5*IQR,
#then it is been defined as outlier
((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
mask

# now replacing all the outliers using the median of that particular column
for i in mask.columns:
    df[i].astype('float')
    temp = df[i].median()
    df.loc[mask[i], i] = temp

# outliers are now being handled and are replaced with that column's median value
((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

# Plot box plots for summary statistics of numerical columns
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),cmap='YlGnBu',annot=True)
plt.show()

# choosing features and target variable
X = df[['CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)']]
y = df['C6H6(GT)']

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Splitting the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

y_train

# You have to modify the model as instructed in the instructions.
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 64  # adjust as needed
model = ANN(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model with validation
num_epochs = 100 
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Validation
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

  # Evaluation
with torch.no_grad():
    y_pred_train = model(X_train)
    train_rmse = torch.sqrt(criterion(y_pred_train, y_train)).item()
    train_r2 = r2_score(y_train.numpy(), y_pred_train.numpy())

    y_pred_test = model(X_test)
    test_rmse = torch.sqrt(criterion(y_pred_test, y_test)).item()
    test_r2 = r2_score(y_test.numpy(), y_pred_test.numpy())

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R^2 Score:", train_r2)
print("Test R^2 Score:", test_r2)
