import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json

from mlflow.models.signature import infer_signature
mlflow.set_tracking_uri("file:./mlruns")
# =========================
# 1. Chargement data
# =========================
df = pd.read_csv("hr_data.csv")

X = df.drop(columns=["churn"])
y = df["churn"]

# =========================
# 2. Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. Model
# =========================
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# =========================
# 4. Predictions
# =========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# =========================
# 5. Metrics
# =========================
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# =========================
# 6. MLflow setup
# =========================
mlflow.set_experiment("hr_churn_experiment")

with mlflow.start_run():

    # Params
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 50)

    # Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)

    # =========================
    # 7. Signature (IMPORTANT FIX)
    # =========================
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.head(1)

    # =========================
    # 8. Log model (FIXED)
    # =========================
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=input_example
    )

    # =========================
    # 9. Schema (monitoring)
    # =========================
    schema = {
        "columns": list(X.columns),
        "dtypes": {col: str(X[col].dtype) for col in X.columns}
    }

    with open("data_schema.json", "w") as f:
        json.dump(schema, f)

    mlflow.log_artifact("data_schema.json")

    # =========================
    # 10. Reference data (drift)
    # =========================
    df.to_csv("reference_data.csv", index=False)
    mlflow.log_artifact("reference_data.csv")

# =========================
# 11. Output
# =========================
print(f"Accuracy: {acc:.2f}, AUC: {auc:.2f}")
print("Modèle loggué dans MLflow avec signature ✔")