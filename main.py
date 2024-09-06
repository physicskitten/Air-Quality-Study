import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
# The dataset is loaded with a semicolon separator and comma as the decimal point.
df = pd.read_csv("USB/Air-Quality-Study/AirQuality.csv", sep=";", decimal=",")
print("First few rows of the dataset:")
print(df.head())

# Visualize missing data using a heatmap
# This helps in understanding the extent and pattern of missing data in the dataset.
plt.figure(figsize=(12, 6))
sns.heatmap(df.isna(), yticklabels=False, cmap='crest', cbar=False)
plt.title("Heatmap of Missing Data")
plt.show()

# Remove unnecessary columns (e.g., unnamed index columns)
# Unnamed columns are usually generated when the index is stored as a column during CSV creation.
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Dataset after removing unnecessary columns:")
print(df.head())

# Replace -200 values with NaN (missing data)
# -200 is used in this dataset to represent missing or invalid data.
df.replace(to_replace=-200, value=np.nan, inplace=True)

# Drop rows with missing data
# This is a straightforward approach to handle missing data, but might result in data loss.
# Consider using imputation techniques if too much data is being dropped.
df.dropna(inplace=True)
print("Dataset after dropping rows with missing data:")
print(df.tail())

# Feature Engineering: Extract 'Month' from 'Date' if applicable
# This helps in analyzing seasonal trends in air quality.
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month

# Visualization: Histogram of CO(GT)
# Visualize the distribution of CO levels to understand its frequency and range.
plt.figure(figsize=(10, 6))
df['CO(GT)'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of CO(GT)')
plt.xlabel('CO(GT)')
plt.ylabel('Frequency')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Visualization: Average CO(GT) by Month
# Analyze how CO levels vary month by month to check for seasonal patterns.
monthly_avg = df.groupby('Month')['CO(GT)'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette='crest')
plt.title('Average CO(GT) by Month')
plt.xlabel('Month')
plt.ylabel('Average CO(GT)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Additional Data Exploration: Summary statistics
# Provides insights into central tendency, dispersion, and overall data distribution.
print("Summary statistics of the dataset:")
print(df.describe())

# Impute missing values with column mean
# Instead of dropping rows, missing data is replaced with the mean of the respective column.
for col in df.columns:
    df[col] = df[col].fillna(df[col].mean())

print("Missing values after imputation:")
print(df.isna().sum())

# Box Plot for Outliers
# Visualize outliers in the data. Outliers are potential anomalies and can affect model performance.
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.title('Box Plot for Outliers checking')
plt.xticks(rotation=45)
plt.show()

# Calculate Interquartile Range (IQR) for outlier detection
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Detect outliers beyond 1.5 * IQR from Q1 and Q3
outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Replace outliers with the median of the respective column
for col in outlier_mask.columns:
    median = df[col].median()
    df.loc[outlier_mask[col], col] = median

# Verify outliers handling
print("Outliers after replacement:")
print(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum())

# Box Plot after handling outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.title('Box Plot after Outlier Handling')
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
# Visualize correlations between features to identify potential relationships.
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)
plt.title('Correlation Matrix')
plt.show()

# Feature Selection and Target Variable
# Select features (independent variables) and the target variable for model training.
X = df[['CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']]
y = df['C6H6(GT)']

# Convert the features and target variable to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Splitting the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define an Artificial Neural Network (ANN) model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation function
        x = self.fc2(x)  # Output layer
        return x

# Initialize the model
input_size = X_train.shape[1]  # Number of input features
hidden_size = 64  # Number of neurons in the hidden layer
model = ANN(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training the model with validation
num_epochs = 100 
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Validation pass
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluation on Training and Test Data
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

# Convert 'Date' and 'Time' columns to a single datetime column if not already done
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')

# Extract various time-based features
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if weekend else 0

#Adding Time Based Feature Extraction for fun, hehe
# Define seasons based on month (Northern Hemisphere season definitions)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)

print("Extracted time-based features:")
print(df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'Weekend', 'Season']].head())

# Create lag features for pollutants and weather data (example with CO(GT))
# Lagging by 1 hour, 2 hours, etc.
lags = [1, 2, 3, 6, 12, 24]  # Define your lag intervals

for lag in lags:
    df[f'CO(GT)_lag_{lag}'] = df['CO(GT)'].shift(lag)
    df[f'NO2(GT)_lag_{lag}'] = df['NO2(GT)'].shift(lag)
    # Add more lagged features as needed

# Drop rows with NaN values generated by lagging
df.dropna(inplace=True)

print("Lag features added:")
print(df.head())
