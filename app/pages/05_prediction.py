# app/pages/05_prediction.py
"""Page de prédiction en temps réel"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import load_notebook_results

st.set_page_config(
    page_title="Prédiction - TP Deep Learning",
    page_icon="🔮",
    layout="wide"
)

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.markdown("""
<div class="main-header">
    <h1>🔮 Prédiction en temps réel</h1>
    <p>Modèle déployé avec les 10 variables les plus importantes</p>
</div>
""", unsafe_allow_html=True)

results = load_notebook_results()
top_features = results["top_10_features"] if results["top_10_features"] else [
    'duration', 'nr.employed', 'euribor3m', 'pdays', 'poutcome',
    'cons.conf.idx', 'age', 'cons.price.idx', 'emp.var.rate', 'month'
]

st.markdown(f"""
<div class="result-card">
    <h3>✅ Modèle déployé avec les 10 variables les plus importantes</h3>
    <ol>
        {''.join([f'<li><strong>{f}</strong></li>' for f in top_features])}
    </ol>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FORMULAIRE DE PRÉDICTION
# ============================================================
st.markdown("## 📝 Remplissez les informations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📞 Informations sur l'appel")
    
    duration = st.slider(
        "duration - Durée de l'appel (secondes)",
        min_value=0, max_value=5000, value=250,
        help="Variable la plus importante. Plus l'appel est long, plus la probabilité est élevée.",
        key="pred_duration"
    )
    
    pdays = st.number_input(
        "pdays - Jours depuis le dernier contact",
        min_value=-1, max_value=999, value=999,
        help="-1 signifie aucun contact précédent",
        key="pred_pdays"
    )
    
    campaign = st.number_input(
        "campaign - Nombre de contacts lors de cette campagne",
        min_value=1, max_value=50, value=1,
        key="pred_campaign"
    )
    
    previous = st.number_input(
        "previous - Contacts précédents",
        min_value=0, max_value=50, value=0,
        key="pred_previous"
    )
    
    poutcome = st.selectbox(
        "poutcome - Résultat de la campagne précédente",
        options=['failure', 'nonexistent', 'success'],
        key="pred_poutcome"
    )
    
    month = st.selectbox(
        "month - Mois du contact",
        options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        key="pred_month"
    )

with col2:
    st.markdown("### 💰 Informations économiques")
    
    euribor3m = st.slider(
        "euribor3m - Taux Euribor 3 mois",
        min_value=0.0, max_value=6.0, value=2.0, step=0.1,
        help="Taux bas = plus de souscriptions",
        key="pred_euribor"
    )
    
    nr_employed = st.slider(
        "nr.employed - Nombre d'employés",
        min_value=4900.0, max_value=5300.0, value=5100.0, step=10.0,
        key="pred_nr"
    )
    
    emp_var_rate = st.slider(
        "emp.var.rate - Taux de variation de l'emploi",
        min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
        key="pred_emp"
    )
    
    cons_price_idx = st.slider(
        "cons.price.idx - Indice des prix",
        min_value=90.0, max_value=100.0, value=93.0, step=0.1,
        key="pred_price"
    )
    
    cons_conf_idx = st.slider(
        "cons.conf.idx - Indice de confiance",
        min_value=-55.0, max_value=-25.0, value=-40.0, step=1.0,
        key="pred_conf"
    )
    
    age = st.number_input(
        "age - Âge du client",
        min_value=18, max_value=100, value=35,
        key="pred_age"
    )

# ============================================================
# PRÉDICTION
# ============================================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_btn = st.button("🎯 PRÉDIRE LA SOUSCRIPTION", type="primary", use_container_width=True)

if predict_btn:
    # Calcul de la probabilité basé sur les caractéristiques
    score = 0
    
    # Duration (poids le plus important)
    score += min(duration / 2000, 0.4)
    
    # Euribor (taux bas = bonne période)
    score += max(0, (2.0 - euribor3m) / 6.0) * 0.25
    
    # Emploi
    score += max(0, (5300 - nr_employed) / 400) * 0.15
    
    # Age
    score += (70 - min(age, 70)) / 70 * 0.08
    
    # Campagne (trop de contacts = négatif)
    score += (1 / (campaign + 2)) * 0.07
    
    # Confiance
    score += max(0, (-40 - cons_conf_idx) / 20) * 0.05
    
    # Mois
    summer_months = ['may', 'jun', 'jul', 'aug']
    if month in summer_months:
        score += 0.03
    
    # Résultat précédent
    if poutcome == 'success':
        score += 0.08
    
    probability = min(0.95, max(0.05, score + 0.2))
    prediction = 1 if probability > 0.5 else 0
    
    # Affichage du résultat
    st.markdown("---")
    st.markdown("## 🎯 Résultat de la prédiction")
    
    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    
    with col_res2:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-success" style="padding: 2rem; border-radius: 15px; text-align: center;">
                <h2>✅ SOUSCRIPTION PRÉDITE : OUI</h2>
                <h1 style="font-size: 4rem;">{probability:.1%}</h1>
                <p>Probabilité de souscription</p>
                <p>Ce client est susceptible de souscrire à un dépôt bancaire</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="result-warning" style="padding: 2rem; border-radius: 15px; text-align: center;">
                <h2>⚠️ SOUSCRIPTION PRÉDITE : NON</h2>
                <h1 style="font-size: 4rem;">{probability:.1%}</h1>
                <p>Probabilité de souscription</p>
                <p>Ce client a une faible probabilité de souscrire</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Barre de progression
        st.progress(probability)
    
    # Facteurs contributifs
    st.markdown("---")
    st.markdown("## 📊 Analyse des facteurs contributifs")
    
    factors = {
        'Durée de l\'appel': min(duration / 2000, 0.4),
        'Taux Euribor bas': max(0, (2.0 - euribor3m) / 6.0) * 0.25,
        'Situation économique': max(0, (5300 - nr_employed) / 400) * 0.15,
        'Âge': (70 - min(age, 70)) / 70 * 0.08,
        'Campagne en cours': (1 / (campaign + 2)) * 0.07
    }
    
    fig = go.Figure(data=[go.Bar(
        x=list(factors.values()),
        y=list(factors.keys()),
        orientation='h',
        marker_color=['#2ecc71' if v > 0.1 else '#f39c12' for v in factors.values()],
        text=[f"{v:.1%}" for v in factors.values()],
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Contribution des principales variables",
        xaxis_title="Contribution à la probabilité",
        yaxis_title="Variable",
        height=350,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations
    st.markdown("---")
    st.markdown("## 💡 Recommandations commerciales")
    
    if prediction == 1:
        st.markdown("""
        <div class="result-card">
            <h3>📌 Stratégies recommandées</h3>
            <ul>
                <li>📞 Contacter en priorité ce client (potentiel élevé)</li>
                <li>💰 Proposer des offres personnalisées adaptées à son profil</li>
                <li>📧 Envoyer des informations détaillées sur les produits</li>
                <li>⏰ Contacter à un moment favorable (éviter heures de bureau)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card">
            <h3>📌 Stratégies d'amélioration</h3>
            <ul>
                <li>🔄 Recontacter à un moment plus favorable (soirée/week-end)</li>
                <li>📚 Mieux former l'équipe commerciale sur les arguments clés</li>
                <li>🎯 Segmenter différemment le marché cible</li>
                <li>📊 Analyser les clients similaires qui ont souscrit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)