# src/models.py
"""Module d'entraînement des modèles pour le TP Deep Learning"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score
import joblib
import os

def train_random_forest(X_train, y_train, X_test, y_test, top_10_features=None):
    """
    Entraîne un modèle Random Forest
    
    Args:
        X_train, y_train: données d'entraînement
        X_test, y_test: données de test
        top_10_features: noms des 10 meilleures variables (optionnel)
    
    Returns:
        model, metrics_dict
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU MODÈLE RANDOM FOREST")
    print("="*60)
    
    if top_10_features is not None:
        print(f"📊 Utilisation des 10 variables les plus importantes:")
        for i, f in enumerate(top_10_features, 1):
            print(f"   {i}. {f}")
    
    # Modèle Random Forest optimisé
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Entraînement
    print("\n🏋️ Entraînement en cours...")
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Performances sur le test set:")
    print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC      : {auc:.4f}")
    
    # Rapport détaillé
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non (no)', 'Oui (yes)']))
    
    metrics = {
        'accuracy': accuracy, 
        'f1': f1, 
        'auc': auc,
        'y_pred': y_pred, 
        'y_pred_proba': y_pred_proba
    }
    
    return model, metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test, top_10_features=None):
    """
    Entraîne un modèle Gradient Boosting
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU MODÈLE GRADIENT BOOSTING")
    print("="*60)
    
    if top_10_features is not None:
        print(f"📊 Utilisation des 10 variables les plus importantes:")
        for i, f in enumerate(top_10_features, 1):
            print(f"   {i}. {f}")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    print("\n🏋️ Entraînement en cours...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Performances sur le test set:")
    print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC      : {auc:.4f}")
    
    metrics = {
        'accuracy': accuracy, 
        'f1': f1, 
        'auc': auc,
        'y_pred': y_pred, 
        'y_pred_proba': y_pred_proba
    }
    
    return model, metrics


def train_neural_network(X_train, y_train, X_test, y_test, top_10_features=None):
    """
    Entraîne un réseau de neurones
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU RÉSEAU DE NEURONES")
    print("="*60)
    
    if top_10_features is not None:
        print(f"📊 Utilisation des 10 variables les plus importantes:")
        for i, f in enumerate(top_10_features, 1):
            print(f"   {i}. {f}")
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        verbose=False
    )
    
    print("\n🏋️ Entraînement en cours...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Performances sur le test set:")
    print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC      : {auc:.4f}")
    
    metrics = {
        'accuracy': accuracy, 
        'f1': f1, 
        'auc': auc,
        'y_pred': y_pred, 
        'y_pred_proba': y_pred_proba
    }
    
    return model, metrics


def save_model(model, model_name, path='../models/'):
    """Sauvegarde un modèle"""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f'{path}{model_name}.pkl')
    print(f"✅ Modèle sauvegardé: {path}{model_name}.pkl")


def load_model(model_name, path='../models/'):
    """Charge un modèle"""
    full_path = f'{path}{model_name}.pkl'
    if os.path.exists(full_path):
        return joblib.load(full_path)
    else:
        print(f"❌ Modèle non trouvé: {full_path}")
        return None


if __name__ == "__main__":
    print("Module models.py - Fonctions disponibles:")
    print("  - train_random_forest()")
    print("  - train_gradient_boosting()")
    print("  - train_neural_network()")
    print("  - save_model()")
    print("  - load_model()")