import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os

# Set nama eksperimen (MLflow akan membuatnya jika belum ada)
mlflow.set_experiment("Lung Cancer CI")

print("Memulai run MLflow...")

# Get the active run created by 'mlflow run'
run = mlflow.active_run()
run_id = run.info.run_id
print(f"Run ID: {run_id}")

# --- 1. Muat Data ---
# MLflow run dieksekusi dari root folder MLProject
data_path = "lung_cancer_preprocessed.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data dimuat dan dibagi.")

# --- 2. Hyperparameter Tuning ---
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2']
}
lr = LogisticRegression(max_iter=1000, random_state=42)
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

print("Memulai training (GridSearchCV)...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# --- 3. Manual Logging ---
print("Mencatat parameter dan metrik...")
mlflow.log_params(grid_search.best_params_)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_f1_score", f1)

# --- 4. Log Artefak (Model & Plot) ---
print("Mencatat artefak...")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix (Test Set)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

plot_path = "training_confusion_matrix.png"
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)
os.remove(plot_path)

# Log Model
mlflow.sklearn.log_model(best_model, "model")

print(f"Model tersimpan di run {run_id}")
print("Run MLflow selesai.")
