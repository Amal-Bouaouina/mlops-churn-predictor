import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. Chargement des données
# =========================
try:
    # Données de référence (Gold Standard)
    ref_df = pd.read_csv("reference_data.csv")
    print(f"✓ Données de référence chargées: {len(ref_df)} lignes")
except FileNotFoundError:
    print("❌ reference_data.csv non trouvé")
    exit(1)

try:
    # Données de production (logs API)
    prod_df = pd.read_csv("logs/current_data.csv")
    print(f"✓ Données de production chargées: {len(prod_df)} lignes")
except FileNotFoundError:
    print("⚠️ logs/current_data.csv non trouvé, création de données de test")
    # Créer des données de test si le fichier n'existe pas
    prod_df = pd.DataFrame({
        'satisfaction': [0.2, 0.1, 0.3, 0.15, 0.25],
        'last_evaluation': [0.3, 0.2, 0.4, 0.25, 0.35],
        'projects': [6, 7, 5, 6, 7],
        'monthly_hours': [310, 330, 290, 320, 340],
        'tenure': [1, 1, 2, 1, 1],
        'work_accident': [0, 0, 0, 0, 0],
        'promoted_last_5years': [0, 0, 0, 0, 0],
        'salary_level': [1, 1, 1, 1, 1]
    })
    print(f"✓ Données de test créées: {len(prod_df)} lignes")

# Vérifier que les colonnes correspondent
print(f"\nColones référence: {list(ref_df.columns)}")
print(f"Colonnes production: {list(prod_df.columns)}")

# =========================
# 2. Configuration du mapping (important!)
# =========================
# Définir le mapping des colonnes
column_mapping = ColumnMapping()
column_mapping.numerical_features = [
    'satisfaction', 'last_evaluation', 'projects', 
    'monthly_hours', 'tenure', 'salary_level'
]
column_mapping.categorical_features = [
    'work_accident', 'promoted_last_5years'
]

# =========================
# 3. Création du rapport (sans TargetDrift si pas de target)
# =========================
# Vérifier si la colonne churn existe dans les données de prod
if "churn" in prod_df.columns:
    column_mapping.target = 'churn'
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    print("✓ Rapport avec détection de drift (incluant target)")
else:
    # Pas de target dans les données de production
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    print("✓ Rapport avec détection de drift uniquement (pas de target)")

# =========================
# 4. Exécution du rapport
# =========================
print("\n🔄 Génération du rapport de monitoring...")
try:
    report.run(
        reference_data=ref_df,
        current_data=prod_df,
        column_mapping=column_mapping
    )
    
    # Sauvegarde
    report.save_html("monitoring_report.html")
    print("✓ Rapport généré: monitoring_report.html")
    
    # Afficher un résumé
    print("\n📊 RÉSUMÉ DU DRIFT:")
    result_json = report.as_dict()
    
    if 'data_drift' in result_json:
        drift_summary = result_json['data_drift']['data']
        drifted_features = []
        for feature, details in drift_summary.items():
            if isinstance(details, dict) and details.get('drift_detected', False):
                drifted_features.append(feature)
                print(f"  ⚠️ Drift détecté sur: {feature}")
        
        if not drifted_features:
            print("  ✅ Aucun drift détecté")
    
except Exception as e:
    print(f"❌ Erreur lors de la génération du rapport: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✨ Monitoring terminé")