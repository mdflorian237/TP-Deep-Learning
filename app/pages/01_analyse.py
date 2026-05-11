# app/pages/01_analyse.py
"""Page d'analyse exploratoire des données"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Analyse des données - TP Deep Learning",
    page_icon="📊",
    layout="wide"
)

# Charger le CSS
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.markdown("""
<div class="main-header">
    <h1>📊 Analyse exploratoire des données</h1>
    <p>Bank Telemarketing Dataset - Compréhension et visualisation des données</p>
</div>
""", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    data_paths = [
        '../datasets/bank_additional/bank-additional.csv',
        'datasets/bank_additional/bank-additional.csv'
    ]
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';')
            return df
    return None

df = load_data()

if df is None:
    st.error("❌ Fichier de données non trouvé. Exécutez d'abord les notebooks.")
    st.stop()

# ============================================================
# 1. INFORMATIONS GÉNÉRALES
# ============================================================
st.markdown("## 1. Informations générales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[0]:,}</div>
        <div class="metric-label">Instances</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[1]}</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{numeric_cols}</div>
        <div class="metric-label">Variables numériques</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    cat_cols = df.select_dtypes(include=['object']).shape[1] - 1
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{cat_cols}</div>
        <div class="metric-label">Variables catégorielles</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 2. DISTRIBUTION DES CLASSES
# ============================================================
st.markdown("---")
st.markdown("## 2. Distribution de la variable cible")

col1, col2 = st.columns(2)

with col1:
    target_counts = df['y'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=['Non (no)', 'Oui (yes)'],
        values=target_counts.values,
        hole=0.4,
        marker_colors=['#ff6b6b', '#4ecdc4'],
        textinfo='label+percent',
        textposition='auto'
    )])
    fig.update_layout(
        title="Proportion des classes",
        height=450,
        annotations=[dict(text=f"Total: {len(df):,}", x=0.5, y=0.5, font_size=14, showarrow=False)]
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown(f"""
    <div class="result-card">
        <h3>📊 Statistiques</h3>
        <p><strong>Classe 'yes' (souscripteurs):</strong> {target_counts['yes']/len(df)*100:.1f}% ({target_counts['yes']:,})</p>
        <p><strong>Classe 'no' (non-souscripteurs):</strong> {target_counts['no']/len(df)*100:.1f}% ({target_counts['no']:,})</p>
        <p><strong>Ratio déséquilibre:</strong> 1:{target_counts['no']/target_counts['yes']:.1f}</p>
        <p><strong>Type de problème:</strong> Classification binaire déséquilibrée</p>
        <hr>
        <h4>💡 Métriques recommandées:</h4>
        <ul>
            <li>F1-Score (moyenne harmonique précision/rappel)</li>
            <li>AUC-ROC (performance indépendante du seuil)</li>
            <li>Precision et Recall (plutôt que l'accuracy seule)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 3. TYPES DE VARIABLES
# ============================================================
st.markdown("---")
st.markdown("## 3. Types de variables")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'y' in categorical_cols:
    categorical_cols.remove('y')

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔢 Variables numériques")
    st.markdown(f"**{len(numeric_cols)} variables:**")
    for col in numeric_cols:
        st.markdown(f"- `{col}`")

with col2:
    st.markdown("#### 🏷️ Variables catégorielles")
    st.markdown(f"**{len(categorical_cols)} variables:**")
    for col in categorical_cols:
        st.markdown(f"- `{col}`")

# ============================================================
# 4. STATISTIQUES DESCRIPTIVES
# ============================================================
st.markdown("---")
st.markdown("## 4. Statistiques descriptives")

tab1, tab2 = st.tabs(["Variables numériques", "Variables catégorielles"])

with tab1:
    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

with tab2:
    for col in categorical_cols[:5]:
        st.markdown(f"**{col}:**")
        st.dataframe(df[col].value_counts().reset_index().head(10), use_container_width=True)
        st.markdown("")

# ============================================================
# 5. MATRICE DE CORRÉLATION
# ============================================================
st.markdown("---")
st.markdown("## 5. Matrice de corrélation")

# Créer une version encodée pour la corrélation
df_corr = df.copy()
df_corr['target'] = df_corr['y'].map({'no': 0, 'yes': 1})
corr_matrix = df_corr[numeric_cols + ['target']].corr()

# Heatmap interactive
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmin=-1, zmax=1,
    text=corr_matrix.round(3).values,
    texttemplate='%{text}',
    textfont={"size": 9},
    hoverongaps=False
))

fig.update_layout(
    title="Matrice de corrélations",
    height=700,
    width=800
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. DISTRIBUTION DES VARIABLES IMPORTANTES
# ============================================================
st.markdown("---")
st.markdown("## 6. Distribution des variables importantes")

important_vars = ['duration', 'euribor3m', 'age', 'campaign', 'nr.employed']

for var in important_vars:
    if var in df.columns:
        st.markdown(f"### {var}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['y']=='no'][var],
                name='Non-souscripteurs',
                opacity=0.7,
                marker_color='#ff6b6b',
                nbinsx=50
            ))
            fig.add_trace(go.Histogram(
                x=df[df['y']=='yes'][var],
                name='Souscripteurs',
                opacity=0.7,
                marker_color='#4ecdc4',
                nbinsx=50
            ))
            fig.update_layout(
                title=f"Distribution de {var}",
                xaxis_title=var,
                yaxis_title="Fréquence",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            stats_no = df[df['y']=='no'][var].describe()
            stats_yes = df[df['y']=='yes'][var].describe()
            
            st.markdown(f"""
            <div class="result-card">
                <h4>Statistiques</h4>
                <table style="width:100%">
                    <tr><th></th><th>Non</th><th>Oui</th></tr>
                    <tr><td>Moyenne</td><td>{stats_no['mean']:.1f}</td><td>{stats_yes['mean']:.1f}</td></tr>
                    <tr><td>Médiane</td><td>{stats_no['50%']:.1f}</td><td>{stats_yes['50%']:.1f}</td></tr>
                    <tr><td>Std</td><td>{stats_no['std']:.1f}</td><td>{stats_yes['std']:.1f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# 7. CONCLUSION
# ============================================================
st.markdown("---")
st.markdown("## 7. Conclusion de l'analyse")

st.markdown(f"""
<div class="result-card">
    <h3>📊 Récapitulatif</h3>
    <ul>
        <li><strong>Dataset:</strong> Bank Telemarketing - {df.shape[0]:,} instances, {df.shape[1]} features</li>
        <li><strong>Problème:</strong> Classification binaire déséquilibrée ({target_counts['yes']/len(df)*100:.1f}% de 'yes')</li>
        <li><strong>Variables clés:</strong> duration, nr.employed, euribor3m, pdays, poutcome</li>
        <li><strong>Défi principal:</strong> Détecter les clients potentiels malgré le déséquilibre</li>
    </ul>
    
    <h3>🎯 Stratégie de modélisation</h3>
    <ul>
        <li>Prétraitement: LabelEncoder + StandardScaler</li>
        <li>Métrique principale: F1-Score (pas accuracy)</li>
        <li>Validation: Cross-validation stratifiée</li>
        <li>Modèles: Random Forest, Gradient Boosting, Réseaux de neurones</li>
    </ul>
</div>
""", unsafe_allow_html=True)