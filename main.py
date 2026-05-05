from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib  # Remplacer mlflow
import os
import json
from typing import List

app = FastAPI()

# =========================
# Charger le modèle depuis le fichier local
# =========================
if os.path.exists("model.joblib"):
    model = joblib.load("model.joblib")
    print("✓ Modèle chargé depuis model.joblib")
else:
    print("❌ Fichier model.joblib non trouvé")
    model = None

# Charger le schéma pour validation
if os.path.exists("data_schema.json"):
    with open("data_schema.json", "r") as f:
        schema = json.load(f)
    print("✓ Schéma chargé")
else:
    print("⚠️ Schéma non trouvé")

class EmployeeData(BaseModel):
    satisfaction: float = Field(..., ge=0, le=1)
    last_evaluation: float = Field(..., ge=0, le=1)
    projects: int = Field(..., ge=1, le=10)
    monthly_hours: int = Field(..., ge=0, le=500)
    tenure: int = Field(..., ge=0, le=50)
    work_accident: int = Field(..., ge=0, le=1)
    promoted_last_5years: int = Field(..., ge=0, le=1)
    salary_level: int = Field(..., ge=1, le=3)

@app.post("/predict")
def predict(employee: EmployeeData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
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

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}