# Table of Contents
1. [Introduction](#introduction)
2. [Libraries](#libraries)
3. [Loading Data](#loading-data)
4. [Data Exploration](#data-exploration)
5. [Data Preprocessing](#data-preprocessing)
6. [Handling Duplicates](#handling-duplicates)
7. [Missing Value Analysis](#missing-value-analysis)
8. [Feature Engineering](#feature-engineering)
9. [Label Encoding](#label-encoding)
10. [Scaling](#scaling)
11. [Building the Model](#building-the-model)
12. [Custom Data Transformers](#custom-data-transformers)
13. [Results](#results)

## Introduction
This documentation provides a detailed guide to preprocess, clean, and analyze loan data using Python. The process involves loading the data, exploring the data, handling duplicates, analyzing missing values, performing feature engineering, label encoding, scaling, building a logistic regression model, and creating custom data transformers.

**Author: Nihar Raju**

## Libraries
The following libraries are used in this code:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `sklearn.impute`: For handling missing values.
- `sklearn.preprocessing`: For label encoding and scaling.
- `sklearn.model_selection`: For splitting the data.
- `sklearn.linear_model`: For logistic regression.
- `sklearn.metrics`: For evaluating the model.
- `sklearn.base`: For creating custom transformers.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
```

## Loading Data
The data is loaded from CSV files containing loan data.

```python
# Load the data
raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

# Display the first few rows of the training data
raw_train.head()

# Display the last few rows of the test data
raw_test.tail()

# Display the number of unique values in each column of the training data
raw_train.nunique()

# Display information about the training data
raw_train.info()

# Display the shape of the training data
raw_train.shape
```

## Data Exploration
The data is explored to understand its structure and check for duplicates.

```python
# Copy the data into new DataFrames
train_df = raw_train.copy()
test_df = raw_test.copy()

# Display information about the training data
train_df.info()

# Display the first few rows of the training data
train_df.head()

# Display the last few rows of the test data
test_df.tail()
```

## Data Preprocessing
The data is preprocessed by dropping unnecessary columns and handling duplicates.

```python
# Drop the 'Loan_Status' column from the training data
train_df.drop(columns=['Loan_Status'], inplace=True)

# Drop the 'Loan_ID' column from both training and test data
train_df.drop(columns='Loan_ID', inplace=True)
test_df.drop(columns='Loan_ID', inplace=True)

# Display the columns of the training data
train_df.columns

# Check for duplicates in the training data
train_df.duplicated()
train_df[train_df.duplicated()]

# Check for duplicates in the test data
test_df[test_df.duplicated()]

# Drop duplicates in the test data
test_df.drop_duplicates(inplace=True)
test_df
```

## Missing Value Analysis
Missing values are analyzed and imputed using appropriate strategies.

```python
# Check for missing values in the training data
train_df.isna().sum()

# Define numerical and categorical columns
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

# Impute missing values in categorical columns using the most frequent value
cat_imputer = SimpleImputer(strategy="most_frequent")
cat_imputer.fit(train_df[categorical_columns])
train_df[categorical_columns] = cat_imputer.transform(train_df[categorical_columns])
test_df[categorical_columns] = cat_imputer.transform(test_df[categorical_columns])

# Impute missing values in numerical columns using the mean
num_imputer = SimpleImputer(strategy="mean")
num_imputer.fit(train_df[numerical_columns])
train_df[numerical_columns] = num_imputer.transform(train_df[numerical_columns])
test_df[numerical_columns] = num_imputer.transform(test_df[numerical_columns])

# Check for missing values in the training data
train_df.isna().sum()
```

## Feature Engineering
Feature engineering is performed to create new features.

```python
# Combine 'ApplicantIncome' and 'CoapplicantIncome' into a single feature
train_df['ApplicantIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
test_df['ApplicantIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']

# Drop the 'CoapplicantIncome' column
train_df.drop(columns='CoapplicantIncome', inplace=True)
test_df.drop(columns='CoapplicantIncome', inplace=True)

# Display the first few rows of the training data
train_df.head()

# Display the last few rows of the test data
test_df.tail()
```

## Label Encoding
Label encoding is applied to categorical columns.

```python
# Display the unique values in the 'Dependents' and 'Property_Area' columns
train_df.Dependents.unique()
train_df.Property_Area.unique()

# Apply label encoding to categorical columns
for col in categorical_columns:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])
    test_df[col] = LabelEncoder().fit_transform(test_df[col])

# Display the first few rows of the training data
train_df.head()
```

## Scaling
The data is scaled using MinMaxScaler.

```python
# Remove 'CoapplicantIncome' from the numerical columns list
numerical_columns.remove('CoapplicantIncome')

# Initialize the MinMaxScaler
min_max_scaler = MinMaxScaler()

# Scale the training and test data
train_df = min_max_scaler.fit_transform(train_df)
test_df = min_max_scaler.transform(test_df)
```

## Building the Model
A logistic regression model is built and evaluated.

```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_df, train_y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
log = LogisticRegression()
log.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = log.predict(X_test)

# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred_test)
acc
```

## Custom Data Transformers
Custom data transformers are created to handle missing values.

```python
# Define a base transformer class
class DemoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# Define a custom mean imputer class
class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col], inplace=True)
        return X

# Example usage of the custom mean imputer
df = pd.DataFrame(np.random.randint(0, 100, (10, 2)), columns=['A', 'B'])
df.iloc[1, 0] = np.nan
df.iloc[2, 1] = np.nan
df.iloc[3, 0] = np.nan
df.loc[4, 0] = np.nan
df.loc[5, 1] = np.nan
df
```

## Results
The results include the cleaned and preprocessed DataFrame, the logistic regression model's accuracy, and the custom data transformers.

### Cleaned and Preprocessed DataFrame
- **Shape**: The shape of the DataFrame after removing duplicates and handling missing values.
- **Data Types**: The data types of the columns.
- **Missing Values**: The number of missing values in each column.
- **Date and Time Conversion**: The 'Pickup_date' column converted to datetime format.

### Logistic Regression Model
- **Accuracy**: The accuracy of the logistic regression model.

### Custom Data Transformers
- **DemoTransformer**: A base transformer class.
- **MeanImputer**: A custom mean imputer class.



**Author: Nihar Raju**
