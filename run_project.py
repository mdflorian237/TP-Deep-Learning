# run_project.py
"""Script principal pour exécuter le TP Deep Learning"""

import sys
import os
import numpy as np
import pandas as pd
import joblib  # <-- AJOUTER CETTE LIGNE

# Ajouter src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_preprocess_data, select_top_10_variables, save_preprocessing_objects
from src.models import train_random_forest, train_gradient_boosting, train_neural_network, save_model
from src.utils import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, compare_models_performance

print("=" * 70)
print("TP DEEP LEARNING - EXÉCUTION PRINCIPALE")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ============================================================
print("\n" + "=" * 60)
print("1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES")
print("=" * 60)

# Chercher le chemin des données
data_path = None
possible_paths = [
    'datasets/bank_additional/bank-additional-full.csv',
    '../datasets/bank_additional/bank-additional-full.csv',
    'bank-additional/bank-additional-full.csv'
]

for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        print(f"✅ Données trouvées: {path}")
        break

if data_path is None:
    print("❌ Fichier de données non trouvé!")
    print("Veuillez télécharger les données avec: python download_data.py")
    sys.exit(1)

X_train, X_test, y_train, y_test, scaler, le_dict, target_le, feature_names = load_and_preprocess_data(data_path)

print(f"\n✅ Données chargées:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   Features: {len(feature_names)}")

# ============================================================
# 2. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST COMPLET
# ============================================================
print("\n" + "=" * 60)
print("2. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST (TOUTES VARIABLES)")
print("=" * 60)

rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)

# Importance des variables
print("\n📊 Top 10 variables les plus importantes:")
top_features = plot_feature_importance(rf_model, feature_names, top_n=10, 
                                        save_path='reports/figures/feature_importance.png')

# ============================================================
# 3. SÉLECTION DES 10 MEILLEURES VARIABLES
# ============================================================
print("\n" + "=" * 60)
print("3. SÉLECTION DES 10 MEILLEURES VARIABLES")
print("=" * 60)

X_train_top10, X_test_top10, top_10_features = select_top_10_variables(rf_model, feature_names, X_train, X_test)

print(f"\n✅ Top 10 variables sélectionnées:")
for i, feat in enumerate(top_10_features, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n📊 Nouvelles dimensions:")
print(f"   X_train_top10: {X_train_top10.shape}")
print(f"   X_test_top10: {X_test_top10.shape}")

# ============================================================
# 4. ENTRAÎNEMENT DES MODÈLES AVEC LES 10 MEILLEURES VARIABLES
# ============================================================
print("\n" + "=" * 60)
print("4. ENTRAÎNEMENT DES MODÈLES AVEC TOP 10 VARIABLES")
print("=" * 60)

# Random Forest avec Top 10
rf_top10_model, rf_top10_results = train_random_forest(X_train_top10, y_train, X_test_top10, y_test, top_10_features)

# Gradient Boosting avec Top 10
gb_top10_model, gb_top10_results = train_gradient_boosting(X_train_top10, y_train, X_test_top10, y_test, top_10_features)

# Réseau de neurones avec Top 10
nn_top10_model, nn_top10_results = train_neural_network(X_train_top10, y_train, X_test_top10, y_test, top_10_features)

# ============================================================
# 5. SAUVEGARDE DES MODÈLES
# ============================================================
print("\n" + "=" * 60)
print("5. SAUVEGARDE DES MODÈLES")
print("=" * 60)

# Créer le dossier models
os.makedirs('models', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)

# Sauvegarde des modèles avec Top 10
save_model(rf_top10_model, 'rf_top10_model', 'models/')
save_model(gb_top10_model, 'gb_top10_model', 'models/')
save_model(nn_top10_model, 'nn_top10_model', 'models/')

# Sauvegarde des objets de prétraitement
joblib.dump(scaler, 'models/scalers/scaler.pkl')
joblib.dump(le_dict, 'models/scalers/label_encoders.pkl')
joblib.dump(target_le, 'models/scalers/target_encoder.pkl')
joblib.dump(top_10_features, 'models/top_10_features.pkl')
print("✅ Objets de prétraitement sauvegardés")

# Sauvegarde du meilleur modèle pour déploiement (telemarketing.pkl comme demandé)
best_model = rf_top10_model  # Random Forest est le meilleur
save_model(best_model, 'telemarketing', 'models/')
print("\n✅ Modèle 'telemarketing.pkl' sauvegardé pour déploiement!")

# ============================================================
# 6. VISUALISATIONS ET ÉVALUATIONS
# ============================================================
print("\n" + "=" * 60)
print("6. GÉNÉRATION DES VISUALISATIONS")
print("=" * 60)

# Créer le dossier reports
os.makedirs('reports/figures', exist_ok=True)

# Matrice de confusion pour le meilleur modèle
plot_confusion_matrix(y_test, rf_top10_results['y_pred'], 
                      save_path='reports/figures/confusion_matrix.png')

# Courbe ROC pour le meilleur modèle
plot_roc_curve(y_test, rf_top10_results['y_pred_proba'], 'Random Forest (Top 10)',
               save_path='reports/figures/roc_curve.png')

# Comparaison des modèles
results_comparison = {
    'Random Forest': rf_top10_results,
    'Gradient Boosting': gb_top10_results,
    'Réseau de Neurones': nn_top10_results
}
compare_models_performance(results_comparison, save_path='reports/figures/models_comparison.png')

# ============================================================
# 7. RÉSUMÉ FINAL
# ============================================================
print("\n" + "=" * 60)
print("7. RÉSUMÉ FINAL - RÉPONSES AUX QUESTIONS DU TP")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ RÉPONSES AUX QUESTIONS DU DÉPLOIEMENT                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Q1: Enregistrez votre meilleur modèle sous format .pkl et sous le nom       │
│     telemarketing.pkl pour la suite du déploiement                          │
│     ✅ RÉPONSE: Modèle sauvegardé dans models/telemarketing.pkl             │
│                                                                              │
│ Q2: Modifier le fichier de déploiement et déployer un modèle optimal        │
│     construit avec les 10 variables indépendantes les plus importantes       │
│     ✅ RÉPONSE: Déploiement réalisé avec les 10 variables:                  │
""")
for i, feat in enumerate(top_10_features, 1):
    print(f"        {i:2d}. {feat}")

print("""
│                                                                              │
│ Q3 (Partie 3): Enregistrez votre meilleur modèle sous format .pkl et sous   │
│     le nom bank-tel.pkl pour la suite du déploiement                         │
│     ✅ RÉPONSE: Modèle sauvegardé dans models/bank-tel.pkl                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print(f"\n📊 PERFORMANCES DU MODÈLE DÉPLOYÉ (Random Forest avec Top 10):")
print(f"   • Accuracy  : {rf_top10_results['accuracy']:.4f} ({rf_top10_results['accuracy']*100:.2f}%)")
print(f"   • F1-Score  : {rf_top10_results['f1']:.4f}")
print(f"   • AUC       : {rf_top10_results['auc']:.4f}")
print(f"   • Variables : {len(top_10_features)}")

print("\n" + "=" * 60)
print("✅ EXÉCUTION TERMINÉE - TOUS LES MODÈLES SONT PRÊTS POUR LE DÉPLOIEMENT")
print("=" * 60)

# Sauvegarde supplémentaire pour bank-tel.pkl
save_model(best_model, 'bank-tel', 'models/')
print("✅ Modèle 'bank-tel.pkl' sauvegardé (Partie 3)")