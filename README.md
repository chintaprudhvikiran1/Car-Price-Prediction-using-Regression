# Car Price Prediction using Regression

## Overview

This project aims to predict car prices using regression techniques. The dataset undergoes preprocessing, feature engineering, and model training to ensure accurate predictions.
### Imported Relevant Libraries

import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set()

### Dataset
•	Loaded the raw dataset containing car details and prices.

•	Cleaned and preprocessed the data for better model performance.

### Preprocessing
•	Used .describe() to understand the dataset statistics.

•	Dropped irrelevant columns to focus on significant variables.

•	Dealt with missing values by removing rows containing them.

### Exploratory Data Analysis
•	Explored the price distribution using seaborn visualization (sns.distplot).

•	Identified key trends and patterns in the data.

### Outlier Handling
•	Removed outliers by keeping data within the 1st and 99th percentiles for relevant columns.

### OLS Assumptions Check
•	Verified that the dataset satisfies Ordinary Least Squares (OLS) regression assumptions.

### Feature Engineering
•	Converted categorical variables into numerical using dummy variables.

•	Created log_price to normalize the target variable.

•	Separated input features and target variable (log_price).

### Data Splitting
•	Standardized the inputs using StandardScaler.

•	Split data into training and testing sets.

### Model Training
•	Built a regression model.

•	Determined weights (coefficients) and bias.

### Evaluation
•	Tested the model on the test dataset.

•	Calculated residuals and percentage difference between predicted and actual values.

### Conclusion
•	Ensured the model is appropriate for this linear dataset.

•	Validated assumptions and confirmed reliability.



