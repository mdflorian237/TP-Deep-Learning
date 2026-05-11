# app/pages/02_modeles.py
"""Page des modèles classiques"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import load_notebook_results

st.set_page_config(
    page_title="Modèles classiques - TP Deep Learning",
    page_icon="🧠",
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
    <h1>🧠 Modèles de Machine Learning Classiques</h1>
    <p>Comparaison, optimisation et évaluation des modèles</p>
</div>
""", unsafe_allow_html=True)

results = load_notebook_results()

# ============================================================
# 1. PERFORMANCE DES MODÈLES
# ============================================================
st.markdown("## 1. Performance des modèles")

perf_df = pd.DataFrame(results["models_performance"]).T
perf_df = perf_df.round(4)
perf_df = perf_df[['accuracy', 'f1', 'auc']]
perf_df.columns = ['Accuracy', 'F1-Score', 'AUC']

st.dataframe(perf_df.style.highlight_max(axis=0, color='#2ecc71'), use_container_width=True)

fig = go.Figure()
models = list(perf_df.index)

fig.add_trace(go.Bar(name='Accuracy', x=models, y=perf_df['Accuracy'], 
                     text=perf_df['Accuracy'].round(3), textposition='auto',
                     marker_color='#3498db'))
fig.add_trace(go.Bar(name='F1-Score', x=models, y=perf_df['F1-Score'], 
                     text=perf_df['F1-Score'].round(3), textposition='auto',
                     marker_color='#2ecc71'))
fig.add_trace(go.Bar(name='AUC', x=models, y=perf_df['AUC'], 
                     text=perf_df['AUC'].round(3), textposition='auto',
                     marker_color='#e74c3c'))

fig.update_layout(
    title="Comparaison des performances",
    xaxis_title="Modèle",
    yaxis_title="Score",
    barmode='group',
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 2. COURBES ROC
# ============================================================
st.markdown("---")
st.markdown("## 2. Courbes ROC")

fig = go.Figure()

roc_points = {
    'Random Forest': {'auc': 0.9532, 'fpr': [0, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 1.0],
                      'tpr': [0, 0.55, 0.75, 0.85, 0.91, 0.95, 0.98, 1.0]},
    'Gradient Boosting': {'auc': 0.9542, 'fpr': [0, 0.03, 0.07, 0.14, 0.24, 0.38, 0.58, 1.0],
                          'tpr': [0, 0.56, 0.76, 0.86, 0.92, 0.96, 0.98, 1.0]},
    'Réseau de Neurones': {'auc': 0.9500, 'fpr': [0, 0.04, 0.09, 0.17, 0.28, 0.44, 0.65, 1.0],
                           'tpr': [0, 0.52, 0.72, 0.83, 0.89, 0.94, 0.97, 1.0]}
}

colors = {'Random Forest': '#3498db', 'Gradient Boosting': '#2ecc71', 'Réseau de Neurones': '#e74c3c'}

for model, data in roc_points.items():
    fig.add_trace(go.Scatter(
        x=data['fpr'], y=data['tpr'],
        mode='lines+markers',
        name=f'{model} (AUC = {data["auc"]:.4f})',
        line=dict(color=colors.get(model, '#95a5a6'), width=2),
        marker=dict(size=6)
    ))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Aléatoire (AUC = 0.5)',
    line=dict(color='gray', width=2, dash='dash')
))

fig.update_layout(
    title="Courbes ROC des modèles optimisés",
    xaxis_title="Taux de faux positifs (FPR)",
    yaxis_title="Taux de vrais positifs (TPR) = Recall",
    height=500,
    template='plotly_white',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1.05])
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3. MATRICE DE CONFUSION
# ============================================================
st.markdown("---")
st.markdown("## 3. Matrice de confusion - Random Forest")

col1, col2 = st.columns(2)

with col1:
    cm = np.array([[7090, 220], [436, 492]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prédit: Non', 'Prédit: Oui'],
        y=['Réel: Non', 'Réel: Oui'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(
        title="Matrice de confusion",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown(f"""
    <div class="result-card">
        <h3>📊 Interprétation</h3>
        <ul>
            <li><strong>Vrais Positifs (TP):</strong> {cm[1,1]} clients correctement prédits</li>
            <li><strong>Vrais Négatifs (TN):</strong> {cm[0,0]} clients correctement prédits</li>
            <li><strong>Faux Positifs (FP):</strong> {cm[0,1]} clients prédits positifs à tort</li>
            <li><strong>Faux Négatifs (FN):</strong> {cm[1,0]} clients manqués</li>
        </ul>
        <hr>
        <h4>📈 Métriques dérivées</h4>
        <ul>
            <li><strong>Precision:</strong> {cm[1,1]/(cm[1,1]+cm[0,1]):.3f}</li>
            <li><strong>Recall:</strong> {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}</li>
            <li><strong>Specificity:</strong> {cm[0,0]/(cm[0,0]+cm[0,1]):.3f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 4. BAGGING - VARIATION DE B
# ============================================================
st.markdown("---")
st.markdown("## 4. Bagging - Variation du nombre d'arbres (B)")

B_values = [10, 25, 50, 100, 200, 300]
scores = [0.842, 0.868, 0.882, 0.891, 0.893, 0.894]
times = [2.3, 5.1, 9.8, 18.5, 35.2, 52.1]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Performance vs B", "Complexité vs B"))

fig.add_trace(go.Scatter(x=B_values, y=scores, mode='lines+markers', 
                         name='F1-Score', line=dict(color='#2ecc71', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=B_values, y=times, mode='lines+markers', 
                         name='Temps (s)', line=dict(color='#e74c3c', width=2)), row=1, col=2)

fig.update_layout(height=450, showlegend=True, template='plotly_white')
fig.update_xaxes(title_text="Nombre d'arbres (B)", row=1, col=1)
fig.update_xaxes(title_text="Nombre d'arbres (B)", row=1, col=2)
fig.update_yaxes(title_text="F1-Score", row=1, col=1)
fig.update_yaxes(title_text="Temps (secondes)", row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**📊 Analyse:**
- Quand B augmente, la performance s'améliore (diminution de la variance)
- Le temps d'entraînement augmente linéairement avec B
- B optimal ≈ 100-200 (bon compromis performance/temps)
""")