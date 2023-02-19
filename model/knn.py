import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# This import is to find the best hyper parameters to fit the KNN model
from sklearn.model_selection import GridSearchCV

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

# Add columns to the dataset
data.columns = cols

# Shuffle the DataFrame
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

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

# Normalize the feature values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # check the shapes of the resulting arrays
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")
#

# -- Extra
print("-- Running grid search now --")

# define the hyper parameters to search over
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# create a KNeighborsClassifier object
knn = KNeighborsClassifier()

# create a GridSearchCV object
grid_search = GridSearchCV(
    knn,  # estimator
    param_grid,  # hyperparameter space
    cv=5,  # number of folds for cross-validation
    scoring='accuracy',  # metric to optimize over
    n_jobs=-1  # use all available CPU cores
)

# fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# print the best hyper parameters and corresponding score
print("-- Grid search completed and found the best below hyper parameters -- ")
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Create a KNN classifier
model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                             weights=grid_search.best_params_['weights'],
                             metric=grid_search.best_params_['metric'])

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = model.score(X_test, y_test)

print("Model accuracy:", accuracy)
