import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Store the dataset into a variable
data = pd.read_csv('spambase/spambase.data', header=None)

# print(data.head())
# print(data.shape)

# create a dataframe with all training data except the target column
X = data.iloc[:, :-1]

# check that the target variable has been removed
# print(X.head())

# separate target values
y = data.iloc[:, -1]

# view target values
# print(y[0:5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check the shapes of the resulting arrays
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")