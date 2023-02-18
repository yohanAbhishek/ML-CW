import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Store the dataset into a variable
data = pd.read_csv('spambase/spambase.data', header=None)

print(data.head())
print(data.shape)
