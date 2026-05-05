from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import json
from typing import List

app = FastAPI()

# ✅ عرف الفنكسيون قبل
def get_latest_run_id():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("hr_churn_experiment")
    runs = client.search_runs(
        exp.experiment_id,
        order_by=["start_time DESC"],
        max_results=1
    )
    return runs[0].info.run_id

model_uri = "runs:/" + get_latest_run_id() + "/model"
model = mlflow.sklearn.load_model(model_uri)
# Charger le schéma pour validation
with open("data_schema.json", "r") as f:
    schema = json.load(f)

class EmployeeData(BaseModel):
    satisfaction: float = Field(..., ge=0, le=1)
    last_evaluation: float = Field(..., ge=0, le=1)
    projects: int = Field(..., ge=1, le=10)
    monthly_hours: int = Field(..., ge=0, le=500)
    tenure: int = Field(..., ge=0, le=50)
    work_accident: int = Field(..., ge=0, le=1)
    promoted_last_5years: int = Field(..., ge=0, le=1)
    salary_level: int = Field(..., ge=1, le=3)

def get_latest_run_id():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("hr_churn_experiment")
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
    return runs[0].info.run_id

@app.post("/predict")
def predict(employee: EmployeeData):
    # Convertir en DataFrame
    input_df = pd.DataFrame([employee.dict()])

    # Log des données reçues (pour monitoring)
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/current_data.csv"
    if not os.path.isfile(log_file):
        input_df.to_csv(log_file, index=False)
    else:
        input_df.to_csv(log_file, mode='a', header=False, index=False)

    # Prédiction
    proba = model.predict_proba(input_df)[0][1]
    return {"churn_probability": round(proba, 4)}