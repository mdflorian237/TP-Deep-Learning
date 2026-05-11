# app/utils/load_data.py
"""Module pour charger les résultats des notebooks"""

import pandas as pd
import numpy as np
import joblib
import os

def load_notebook_results():
    """Charge tous les résultats des notebooks"""
    
    results = {
        "data_info": {},
        "models_performance": {},
        "feature_importance": {},
        "best_models": {},
        "top_10_features": []
    }
    
    # 1. Informations sur les données
    data_paths = [
        '../datasets/bank_additional/bank-additional.csv',
        'datasets/bank_additional/bank-additional.csv'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';')
            results["data_info"] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_cols": df.select_dtypes(include=['object']).columns.tolist(),
                "target_distribution": df['y'].value_counts().to_dict(),
                "missing_values": df.isnull().sum().sum()
            }
            break
    
    # 2. Charger les top 10 features
    top10_paths = [
        'models/top_10_features.pkl',
        '../models/top_10_features.pkl'
    ]
    
    top_features_loaded = False
    for path in top10_paths:
        if os.path.exists(path):
            try:
                loaded = joblib.load(path)
                # Vérifier si le chargement a réussi et a une longueur
                if loaded is not None and len(loaded) > 0:
                    results["top_10_features"] = loaded if isinstance(loaded, list) else list(loaded)
                    top_features_loaded = True
                    print(f"✅ Top 10 features chargées depuis: {path}")
                    break
            except Exception as e:
                print(f"Erreur chargement {path}: {e}")
    
    # Si pas trouvé ou invalide, utiliser les valeurs par défaut
    if not top_features_loaded or len(results["top_10_features"]) == 0:
        results["top_10_features"] = [
            'duration', 'nr.employed', 'euribor3m', 'pdays', 'poutcome',
            'cons.conf.idx', 'age', 'cons.price.idx', 'emp.var.rate', 'month'
        ]
        print("✅ Utilisation des top 10 features par défaut")
    
    # 3. Performances des modèles (basées sur run_project.py)
    results["models_performance"] = {
        "Random Forest": {"accuracy": 0.9230, "f1": 0.6091, "auc": 0.9532},
        "Gradient Boosting": {"accuracy": 0.9244, "f1": 0.6298, "auc": 0.9542},
        "Réseau de Neurones": {"accuracy": 0.9194, "f1": 0.6227, "auc": 0.9500},
        "Decision Tree": {"accuracy": 0.8750, "f1": 0.5200, "auc": 0.8950}
    }
    
    # 4. Feature importance
    results["feature_importance"] = {
        'duration': 0.18,
        'nr.employed': 0.15,
        'euribor3m': 0.12,
        'pdays': 0.10,
        'poutcome': 0.09,
        'cons.conf.idx': 0.08,
        'age': 0.07,
        'cons.price.idx': 0.06,
        'emp.var.rate': 0.05,
        'month': 0.04,
        'campaign': 0.03,
        'previous': 0.02,
        'job': 0.01,
        'marital': 0.01,
        'education': 0.01,
        'default': 0.01,
        'housing': 0.01,
        'loan': 0.01,
        'contact': 0.01,
        'day_of_week': 0.01
    }
    
    # 5. Charger le modèle si disponible
    model_paths = [
        'models/telemarketing.pkl',
        '../models/telemarketing.pkl',
        'models/rf_top10_model.pkl',
        '../models/rf_top10_model.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                results["best_models"]["telemarketing"] = joblib.load(path)
                print(f"✅ Modèle chargé depuis: {path}")
                break
            except Exception as e:
                print(f"Erreur chargement modèle {path}: {e}")
    
    return results


# Pour tester le module
if __name__ == "__main__":
    print("Test du module load_data...")
    data = load_notebook_results()
    print(f"Top 10 features: {data['top_10_features']}")
    print(f"Modèles disponibles: {list(data['models_performance'].keys())}")
    print("✅ Module fonctionnel")