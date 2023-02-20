"""
◉ This is a class to run KNN on the spambase dataset.
Additional:
    ◦ In terms of finding the best hyper parameters to use in this model I have implemented a grid search.
    ◦ Saved the model
"""

from sklearn.neighbors import KNeighborsClassifier
import preprocess as p
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def run_grid(classifier):
    print("-- Running grid search now --")

    param_grid = {  # define the hyper parameters to search over
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(  # create a GridSearchCV object
        classifier,  # estimator
        param_grid,  # hyperparameter space
        cv=5,  # number of folds for cross-validation
        scoring='accuracy',  # metric to optimize over
        n_jobs=-1  # use all available CPU cores
    )

    # fit the GridSearchCV object to the training data
    grid_search.fit(p.get_X_train(), p.get_y_train())

    # print the best hyper parameters and corresponding score
    print("-- Grid search completed and found the best below hyper parameters -- ")
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    return grid_search.best_params_


def create_model():
    # Run grid search and get the best parameters
    search = run_grid(KNeighborsClassifier())

    # Create a KNN classifier
    model = KNeighborsClassifier(n_neighbors=search['n_neighbors'],
                                 weights=search['weights'],
                                 metric=search['metric'])

    # Fit the model on the training data
    model.fit(p.get_X_train(), p.get_y_train())

    return model


m = create_model()

# Save the model
# joblib.dump(m, 'knn_model.joblib')

# Make predictions on the test data
y_pred = m.predict(p.get_X_test())

accuracy = accuracy_score(p.get_y_test(), y_pred)
matrix = confusion_matrix(p.get_y_test(), y_pred)
report = classification_report(p.get_y_test(), y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{matrix}")
print(f"Classification Report:\n{report}")
