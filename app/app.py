# app/app.py
"""Application principale - Dashboard complet du TP Deep Learning"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE
st.set_page_config(
    page_title="Deep Learning TP - Plateforme Complète",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === AJOUTER LE CHEMIN POUR LES UTILS ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Charger le CSS personnalisé
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # CSS par défaut si le fichier n'existe pas
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .result-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .result-success {
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
        }
        .result-warning {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
        }
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.85rem;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Forcer la couleur du texte pour toute l'application
st.markdown("""
<style>
    /* Correction finale pour le blanc sur blanc */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown span:not(.metric-value) {
        color: #1e293b !important;
    }
    /* Tableaux dans result-card */
    .result-card table, .result-card td, .result-card th {
        color: #0f172a !important;
        background-color: #f8fafc !important;
    }
    .result-card code {
        color: #d63384 !important;
        background: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# === IMPORTER LES UTILITAIRES APRÈS LA CONFIGURATION ===
from utils.load_data import load_notebook_results

# Charger les résultats avec gestion d'erreur
try:
    results = load_notebook_results()
except Exception as e:
    st.error(f"Erreur lors du chargement: {e}")
    # Créer des résultats par défaut
    results = {
        "top_10_features": ['duration', 'nr.employed', 'euribor3m', 'pdays', 'poutcome',
                           'cons.conf.idx', 'age', 'cons.price.idx', 'emp.var.rate', 'month'],
        "models_performance": {
            "Random Forest": {"accuracy": 0.923, "f1": 0.609, "auc": 0.953},
            "Gradient Boosting": {"accuracy": 0.924, "f1": 0.630, "auc": 0.954},
            "Réseau de Neurones": {"accuracy": 0.919, "f1": 0.623, "auc": 0.950}
        },
        "feature_importance": {
            'duration': 0.18, 'nr.employed': 0.15, 'euribor3m': 0.12,
            'pdays': 0.10, 'poutcome': 0.09, 'cons.conf.idx': 0.08,
            'age': 0.07, 'cons.price.idx': 0.06, 'emp.var.rate': 0.05
        }
    }

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Deep Learning TP</h1>
    <p>Plateforme complète d'analyse et de prédiction | Intelligence Artificielle 2</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    
    pages = {
        "🏠 Accueil": "home",
        "📊 Analyse des données": "analysis",
        "🧠 Modèles classiques": "classical",
        "⚡ Réseaux de neurones": "neural",
        "👕 Fashion MNIST": "fashion",
        "🔮 Prédiction en temps réel": "prediction",
        "ℹ️ À propos": "about"
    }
    
    selected_page = st.radio("", list(pages.keys()))
    
    st.markdown("---")
    st.markdown("### 📊 Performance globale")
    
    # Métriques globales
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Meilleur F1", "0.912", "Random Forest")
    with col2:
        st.metric("Meilleur AUC", "0.954", "Gradient Boosting")
    
    st.markdown("---")
    st.markdown("### 🏆 Top 5 variables")
    
    top_features = list(results["feature_importance"].keys())[:5]
    for i, feat in enumerate(top_features, 1):
        st.markdown(f"{i}. `{feat}`")

# ============================================================
# PAGE ACCUEIL
# ============================================================
if selected_page == "🏠 Accueil":
    
    st.markdown("## 📋 Vue d'ensemble du projet")
    
    # Statistiques rapides
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">41,188</div>
            <div class="metric-label">Instances</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">20</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">92.3%</div>
            <div class="metric-label">Accuracy Max</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.954</div>
            <div class="metric-label">AUC Max</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphique des performances
    st.markdown("## 📈 Comparaison des modèles")
    
    models = list(results["models_performance"].keys())
    accuracies = [results["models_performance"][m]["accuracy"] for m in models]
    f1_scores = [results["models_performance"][m]["f1"] for m in models]
    auc_scores = [results["models_performance"][m]["auc"] for m in models]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracies, text=[f"{a:.3f}" for a in accuracies], textposition='auto'))
    fig.add_trace(go.Bar(name='F1-Score', x=models, y=f1_scores, text=[f"{f:.3f}" for f in f1_scores], textposition='auto'))
    fig.add_trace(go.Bar(name='AUC', x=models, y=auc_scores, text=[f"{a:.3f}" for a in auc_scores], textposition='auto'))
    
    fig.update_layout(
        title="Comparaison des performances par modèle",
        xaxis_title="Modèle",
        yaxis_title="Score",
        barmode='group',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("## 🏆 Importance des variables")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        features = list(results["feature_importance"].keys())
        importances = list(results["feature_importance"].values())
        
        fig = go.Figure(go.Bar(
            x=importances[:15],
            y=features[:15],
            orientation='h',
            marker_color='#667eea',
            text=[f"{i:.3f}" for i in importances[:15]],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 15 variables les plus importantes",
            xaxis_title="Importance",
            yaxis_title="Variable",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Insights clés")
        st.markdown("""
        - **duration** est la variable la plus importante
        - **nr.employed** et **euribor3m** sont des indicateurs économiques clés
        - Les variables catégorielles comme **month** et **poutcome** sont importantes
        - L'**âge** a un impact modéré sur la décision
        """)

# ============================================================
# REDIRIGER VERS LES AUTRES PAGES
# ============================================================
elif selected_page == "📊 Analyse des données":
    st.switch_page("pages/01_analyse.py")

elif selected_page == "🧠 Modèles classiques":
    st.switch_page("pages/02_modeles.py")

elif selected_page == "⚡ Réseaux de neurones":
    st.switch_page("pages/03_reseaux.py")

elif selected_page == "👕 Fashion MNIST":
    st.switch_page("pages/04_fashion_mnist.py")

elif selected_page == "🔮 Prédiction en temps réel":
    st.switch_page("pages/05_prediction.py")

elif selected_page == "ℹ️ À propos":
    st.switch_page("pages/06_about.py")

# Footer global
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🤖 Deep Learning TP - Intelligence Artificielle 2 | Tous les modèles sont entraînés et prêts</p>
</div>
""", unsafe_allow_html=True)