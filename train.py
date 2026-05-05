import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json

# Chargement
df = pd.read_csv("hr_data.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métriques
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# MLflow tracking
mlflow.set_experiment("hr_churn_experiment")
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    # Sauvegarde du modèle
    mlflow.sklearn.log_model(model, "model")

    # Sauvegarde du schéma (nécessaire pour monitoring)
    schema = {
        "columns": list(X.columns),
        "dtypes": {col: str(X[col].dtype) for col in X.columns}
    }
    with open("data_schema.json", "w") as f:
        json.dump(schema, f)
    mlflow.log_artifact("data_schema.json")

    # Sauvegarde de la référence (gold standard)
    df.to_csv("reference_data.csv", index=False)
    mlflow.log_artifact("reference_data.csv")

print(f"Accuracy: {acc:.2f}, AUC: {auc:.2f}")
print("Modèle loggué dans MLflow")