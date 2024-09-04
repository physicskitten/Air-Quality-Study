# Air Quality Study - README

## Overview

This project involves analyzing and modeling air quality data using various data processing, visualization, and machine learning techniques. The goal is to predict the concentration of Benzene (`C6H6(GT)`) in the air based on several sensor readings. The project includes data preprocessing, exploratory data analysis, feature engineering, and the implementation of a simple Artificial Neural Network (ANN) using PyTorch for regression.

## Prerequisites

### Libraries and Dependencies

Ensure that the following Python libraries are installed:

- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and data visualization
- `seaborn`: Statistical data visualization
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning tools
- `torch`: PyTorch library for deep learning

You can install these libraries using pip:

```bash

## Data Description

The dataset used in this project contains air quality measurements from different sensors. It includes features like the concentration of CO, NOx, NO2, and other gases, as well as the Benzene concentration, which is the target variable for prediction.

The dataset is stored in a CSV file (`AirQuality.csv`), with the following characteristics:

- The delimiter used is a semicolon (`;`).
- Decimal points are represented by commas (`,`).
- The dataset contains some missing or invalid data, represented by `-200`.

## Project Structure

### 1. Data Loading and Cleaning

- The dataset is loaded using `pandas`, with appropriate handling of the delimiter and decimal separator.
- Unnecessary columns, such as unnamed index columns, are removed.
- Missing data represented by `-200` is replaced with `NaN`, and rows containing `NaN` values are dropped.
- A heatmap is generated to visualize the distribution of missing data.

### 2. Feature Engineering

- The `Date` column is converted to a datetime format, and the month is extracted as a new feature.
- The data is explored through various visualizations:
  - Histogram of CO levels (`CO(GT)`)
  - Monthly average CO levels
  - Summary statistics of the dataset

### 3. Outlier Detection and Handling

- Outliers are identified using the Interquartile Range (IQR) method.
- Detected outliers are replaced with the median of the respective column to minimize their impact on the model.
- Box plots are generated before and after outlier handling to visualize the changes.

### 4. Correlation Analysis

- A correlation heatmap is created to examine the relationships between different features, which helps in feature selection for the model.

### 5. Feature Selection and Data Splitting

- Features (`X`) are selected based on their relevance to the target variable (`C6H6(GT)`).
- The dataset is split into training, validation, and test sets using `train_test_split` from `scikit-learn`.

### 6. Model Building - Artificial Neural Network (ANN)

- An ANN model is defined using PyTorch, with the following structure:
  - Input layer: Equal to the number of features
  - Hidden layer: 64 neurons with ReLU activation
  - Output layer: Single neuron for regression
- The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer.
- The model is trained for 100 epochs, with the loss printed every 10 epochs for both training and validation sets.

### 7. Model Evaluation

- The model's performance is evaluated using the Root Mean Squared Error (RMSE) and R-squared (R²) metrics on both training and test sets.
- These metrics provide insights into the model's accuracy and goodness-of-fit.

## How to Run the Code

1. Place the `AirQuality.csv` file in the appropriate directory (e.g., `USB/Air-Quality-Study/`).
2. Ensure that all required libraries are installed.
3. Run the Python script in your preferred environment (e.g., Jupyter Notebook, PyCharm, or command line).
4. Observe the output and visualizations to understand the data processing steps, model training, and evaluation results.

## Results

The trained model provides predictions for the Benzene concentration based on the input features. The performance metrics (RMSE and R²) help assess the accuracy of these predictions, guiding potential improvements or adjustments to the model.

## Future Improvements

- Experiment with more complex models (e.g., deeper neural networks, ensemble methods) to improve prediction accuracy.
- Implement more sophisticated imputation methods for missing data.
- Explore feature selection techniques to reduce dimensionality and improve model efficiency.
