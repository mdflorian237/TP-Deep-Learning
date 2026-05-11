# src/utils.py
"""Module utilitaires pour le TP Deep Learning"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

def plot_confusion_matrix(y_test, y_pred, class_names=['Non', 'Oui'], save_path=None):
    """Affiche et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de confusion', fontsize=14, fontweight='bold')
    plt.xlabel('Prédit', fontsize=12)
    plt.ylabel('Réel', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅ Matrice sauvegardée: {save_path}")
    
    plt.show()
    return cm


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """Affiche et sauvegarde la courbe ROC"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
    plt.title(f'Courbe ROC - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅ Courbe ROC sauvegardée: {save_path}")
    
    plt.show()
    return fpr, tpr, roc_auc


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """Affiche l'importance des variables"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices][::-1], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} variables les plus importantes', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
    
    plt.show()
    
    # Retourner les top variables
    top_features = [(feature_names[i], importances[i]) for i in indices]
    return top_features


def compare_models_performance(results_dict, save_path=None):
    """Compare les performances de plusieurs modèles"""
    models = list(results_dict.keys())
    f1_scores = [results_dict[m]['f1'] for m in models]
    accuracies = [results_dict[m]['accuracy'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score', color='#3498db')
    bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='#2ecc71')
    
    ax.set_xlabel('Modèles', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des performances des modèles', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.8, 0.95)
    
    # Ajout des valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅ Comparaison sauvegardée: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Module utils.py - Fonctions disponibles:")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - plot_feature_importance()")
    print("  - compare_models_performance()")