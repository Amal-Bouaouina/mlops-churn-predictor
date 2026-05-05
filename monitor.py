import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# =========================
# 1. Charger les données
# =========================
ref_df = pd.read_csv("reference_data.csv")
prod_df = pd.read_csv("logs/current_data.csv")

# =========================
# 2. Nettoyage (IMPORTANT)
# =========================
ref_df = ref_df.dropna()
prod_df = prod_df.dropna()

# Garder فقط الأعمدة المشتركة
common_cols = ref_df.columns.intersection(prod_df.columns)
ref_df = ref_df[common_cols]
prod_df = prod_df[common_cols]

# =========================
# 3. Gestion target (churn absent en prod)
# =========================
if "churn" not in prod_df.columns:
    prod_df["churn"] = -1
    target_available = False
else:
    target_available = True

# =========================
# 4. Rapport Evidently
# =========================
if target_available:
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
else:
    report = Report(metrics=[
        DataDriftPreset()
    ])

# =========================
# 5. Run report
# =========================
report.run(reference_data=ref_df, current_data=prod_df)

# =========================
# 6. Save report
# =========================
report.save_html("monitoring_report.html")

print("✅ Rapport généré : monitoring_report.html")