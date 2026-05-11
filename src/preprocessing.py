# src/preprocessing.py
"""Module de prétraitement des données pour le TP Deep Learning"""
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_preprocess_data(data_path='../datasets/bank_additional/bank-additional-full.csv'):
    """
    Charge et prétraite les données Bank Telemarketing
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, le_dict, target_le
    """
    # Chargement
    df = pd.read_csv(data_path, sep=';')
    
    # Identification des colonnes
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')
    
    # Encodage des variables catégorielles
    le_dict = {}
    df_processed = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        le_dict[col] = le
    
    # Encodage de la cible
    target_le = LabelEncoder()
    y = target_le.fit_transform(df_processed['y'])
    X = df_processed.drop('y', axis=1)
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_dict, target_le, X.columns


def select_top_10_variables(model, feature_names, X_train, X_test):
    """
    Sélectionne les 10 variables les plus importantes
    
    Returns:
        X_train_top10, X_test_top10, top_10_features
    """
    # Récupérer l'importance des variables
    importances = model.feature_importances_
    
    # Créer un DataFrame pour trier
    feat_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Sélectionner les 10 meilleures
    top_10_features = feat_importance.head(10)['feature'].values
    
    # Indices des colonnes à garder
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    top_10_indices = [feature_to_idx[f] for f in top_10_features]
    
    # Filtrer les données
    X_train_top10 = X_train[:, top_10_indices]
    X_test_top10 = X_test[:, top_10_indices]
    
    return X_train_top10, X_test_top10, top_10_features


def save_preprocessing_objects(scaler, le_dict, target_le, top_10_features, path='../models/'):
    """Sauvegarde les objets de prétraitement"""
    os.makedirs(path, exist_ok=True)
    joblib.dump(scaler, f'{path}scaler.pkl')
    joblib.dump(le_dict, f'{path}label_encoders.pkl')
    joblib.dump(target_le, f'{path}target_encoder.pkl')
    joblib.dump(top_10_features, f'{path}top_10_features.pkl')
    print(f"✅ Objets sauvegardés dans {path}")


if __name__ == "__main__":
    # Test du module
    print("Test du module preprocessing...")
    X_train, X_test, y_train, y_test, scaler, le_dict, target_le, feature_names = load_and_preprocess_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print("✅ Module fonctionnel")