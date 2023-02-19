import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

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

# check the shapes of the resulting arrays
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# from sklearn.model_selection import cross_val_score
#
# # define the range of k values to test
# k_range = list(range(1, 31))
#
# # list to store cross-validation scores for each k
# k_scores = []
#
# # perform k-fold cross-validation for each k value
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#     k_scores.append(scores.mean())
#
# # find the optimal value of k that maximizes cross-validation score
# optimal_k = k_range[k_scores.index(max(k_scores))]
# print("The optimal number of neighbors is", optimal_k)


# Create a KNN classifier with 5 NNs
model = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)
