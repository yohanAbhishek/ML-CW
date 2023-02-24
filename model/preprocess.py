"""
◉ This is a class to run preprocessing on the spambase dataset.
steps:
    1 Add column names to the dataset.
    2 Shuffle the dataset.
    3 There is no missing data.
    4 There are no categorical features to be handled.
    5 Separate the dataset into the training and testing sets.
    6 Reduce the dimensionality of the dataset using PCA.
    7 Handle class imbalance (Not implemented in the code).
    8 Normalize feature values using MinMaxScaler.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Store the dataset into a variable
data = pd.read_csv('../spambase/spambase.data', header=None)

# Specify the dataset columns
cols = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
    'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
    'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
    'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
    'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
    'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
    'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'spam'
]

# Add column names to the dataset
data.columns = cols

# Shuffle the dataset
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

# Separate the dataset into training and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduce the dimensionality of the dataset using PCA
n_components = 44  # 90% components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Normalize feature values
scaler = MinMaxScaler()
X_pca = scaler.fit_transform(X_pca)

# Create a scatter plot of the transformed data with color-coded labels
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set2')
plt.show()

# Compute correlation matrix
corr_matrix = data.corr()

# Plot heatmap
# sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()


def get_X_train():
    return X_train


def get_y_train():
    return y_train


def get_X_test():
    return X_test


def get_y_test():
    return y_test
