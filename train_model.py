import mlflow
import mlflow.xgboost
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Set the tracking URI to the remote MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("xgboost_experiment")

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 3, "learning_rate": 0.1, "n_estimators": 100}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

confusion_matrix_path = "confusion_matrix.png"
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(confusion_matrix_path)
plt.close()

feature_importances = model.feature_importances_
features = data.feature_names
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color="skyblue")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
feature_importance_path = "feature_importance.png"
plt.savefig(feature_importance_path)
plt.close()

# Log experiment
with mlflow.start_run():
    mlflow.log_params(params)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifact(feature_importance_path)

    mlflow.xgboost.log_model(model, artifact_path="model")

    print(f"Model logged with accuracy: {accuracy:.4f}")
    print("Confusion matrix and feature importance plots logged as artifacts.")
