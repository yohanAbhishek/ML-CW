import model.preprocess as p
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the saved KNN model
knn_model = joblib.load('../model/trained_models/knn_model.joblib')

# Make predictions on the test data using the KNN model
y_pred_knn = knn_model.predict(p.get_X_test())

# Calculate the accuracy score, confusion matrix, and classification report for KNN
accuracy_knn = accuracy_score(p.get_y_test(), y_pred_knn)
matrix_knn = confusion_matrix(p.get_y_test(), y_pred_knn)
report_knn = classification_report(p.get_y_test(), y_pred_knn)

# Load the saved Decision Tree model
dt_model = joblib.load('../model/trained_models/dt_model.joblib')

# Make predictions on the test data using the Decision Tree model
y_pred_dt = dt_model.predict(p.get_X_test())

# Calculate the accuracy score, confusion matrix, and classification report for Decision Tree
accuracy_dt = accuracy_score(p.get_y_test(), y_pred_dt)
matrix_dt = confusion_matrix(p.get_y_test(), y_pred_dt)
report_dt = classification_report(p.get_y_test(), y_pred_dt)


# Print out the evaluation metrics for both models
print("KNN Evaluation Metrics:")
print(f"Accuracy: {accuracy_knn}")
print(f"Precision: {classification_report(p.get_y_test(), y_pred_knn, output_dict=True)['1']['precision']}")
print(f"Recall: {classification_report(p.get_y_test(), y_pred_knn, output_dict=True)['1']['recall']}")
print(f"F1-Score: {classification_report(p.get_y_test(), y_pred_knn, output_dict=True)['1']['f1-score']}")
print(f"Confusion Matrix:\n{matrix_knn}")
print(f"Classification Report:\n{report_knn}\n")

print("Decision Tree Evaluation Metrics:")
print(f"Accuracy: {accuracy_dt}")
print(f"Precision: {classification_report(p.get_y_test(), y_pred_dt, output_dict=True)['1']['precision']}")
print(f"Recall: {classification_report(p.get_y_test(), y_pred_dt, output_dict=True)['1']['recall']}")
print(f"F1-Score: {classification_report(p.get_y_test(), y_pred_dt, output_dict=True)['1']['f1-score']}")
print(f"Confusion Matrix:\n{matrix_dt}")
print(f"Classification Report:\n{report_dt}\n")
