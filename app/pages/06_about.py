# app/pages/06_about.py
"""Page À propos du TP"""

import streamlit as st
import sys
import os

st.set_page_config(
    page_title="À propos - TP Deep Learning",
    page_icon="ℹ️",
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
    <h1>ℹ️ À propos du TP</h1>
    <p>Deep Learning - Intelligence Artificielle 2</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INFORMATIONS GÉNÉRALES
# ============================================================
st.markdown("## 🎓 Informations académiques")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="result-card">
        <h3>📚 Cours</h3>
        <ul>
            <li><strong>Intitulé:</strong> Intelligence Artificielle 2: DL</li>
            <li><strong>Niveau:</strong> DIPES 2 - 4ème année (S2)</li>
            <li><strong>Année académique:</strong> 2025-2026</li>
            <li><strong>Enseignant:</strong> Stéphane C. K. TÉKOUABOU (PhD & Ing.)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="result-card">
        <h3>👨‍💻 Auteur</h3>
        <ul>
            <li><strong>Nom:</strong> Florian Dongmo</li>
            <li><strong>Filière:</strong> Intelligence Artificielle</li>
            <li><strong>Date:</strong> Avril 2026</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CONTENU DU TP
# ============================================================
st.markdown("---")
st.markdown("## 📋 Contenu du TP")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Partie 1: Analyse et modèles classiques
    
    **a- Analyse des données**
    - Structure et types de variables
    - Distribution des classes
    - Matrice de corrélations
    - Sélection de variables
    
    **b- Prétraitement**
    - Encodage des variables catégorielles
    - Normalisation (StandardScaler)
    - Split 80%/20%
    
    **c- Modèles simples**
    - SVM, KNN, Decision Tree, NB, LR
    
    **d- Validation croisée**
    - Classifieur constant (baseline)
    - Arbre CART optimisé (GridSearchCV)
    
    **e- Bagging et Random Forest**
    - Impact de B sur les performances
    - Erreur Out-Of-Bag
    - Optimisation de p (max_features)
    
    **f- Boosting**
    - Gradient Boosting
    - Early stopping
    - AdaBoost
    """)

with col2:
    st.markdown("""
    ### Partie 2: Réseaux de neurones
    
    **c- Modélisation**
    - MLPClassifier (scikit-learn)
    - Architecture multi-couches
    - Optimisation des hyperparamètres
    
    ### Partie 3: Deep Learning
    
    **Fashion MNIST**
    - 70,000 images (28x28)
    - 10 classes de vêtements
    - Classification avec MLP et PCA
    - Categorical crossentropy
    
    ### Déploiement
    
    **Application Streamlit**
    - Interface complète et professionnelle
    - Prédiction avec les 10 variables importantes
    - Dashboard de performances
    """)

# ============================================================
# RÉSULTATS OBTENUS
# ============================================================
st.markdown("---")
st.markdown("## 📊 Résultats obtenus")

results_data = {
    "Modèle": ["Random Forest", "Gradient Boosting", "Réseau de Neurones", "MLP (Fashion MNIST)"],
    "Accuracy": ["92.30%", "92.44%", "91.94%", "91.20%"],
    "F1-Score": ["0.609", "0.630", "0.623", "0.909"],
    "AUC": ["0.953", "0.954", "0.950", "N/A"]
}

st.dataframe(results_data, use_container_width=True)

# ============================================================
# RÉPONSES AUX QUESTIONS
# ============================================================
st.markdown("---")
st.markdown("## ✅ Récapitulatif des réponses aux questions")

st.markdown("""
| Question | Réponse | Où ? |
|----------|---------|------|
| Combien de classes ? | 2 (yes/no) | Notebook 01 |
| Type d'apprentissage | Classification binaire supervisée | Notebook 01 |
| Nb caractéristiques | 20 (10 numériques, 10 catégorielles) | Notebook 01 |
| Métriques recommandées | F1-Score, AUC, Precision, Recall | Notebook 01 |
| Matrice de corrélations | duration (+0.41), euribor3m (-0.31) | Notebook 01 |
| Sélection de variables | Oui, top 10 utilisées | run_project.py |
| Meilleur modèle | Random Forest (F1=0.609, AUC=0.953) | Notebook 02 |
| Classifieur constant | Accuracy=88.7%, F1=0 | Notebook 02 |
| Bagging B optimal | B=100-200 | Notebook 02 |
| Feature importance | duration, nr.employed, euribor3m | Notebook 02 |
| Modèle déployé | telemarketing.pkl avec top 10 variables | app/prediction.py |
""")

# ============================================================
# NOTES ET RÉFÉRENCES
# ============================================================
st.markdown("---")
st.markdown("## 📚 Notes et références")

st.markdown("""
### Bibliographie

- UCI Machine Learning Repository: Bank Marketing Dataset
- Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms
- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/

### Technologies utilisées

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| Python | 3.14.1 | Langage principal |
| scikit-learn | 1.8.0 | Modèles ML |
| pandas | 2.3.3 | Manipulation de données |
| numpy | 2.4.1 | Calculs numériques |
| matplotlib | 3.10.8 | Visualisations |
| seaborn | 0.13.2 | Visualisations statistiques |
| plotly | 6.5.2 | Graphiques interactifs |
| streamlit | 1.53.0 | Interface web |

### Structure du projet
TP_Deep_Learning/
├── notebooks/ # 4 notebooks d'analyse
├── src/ # Code source (preprocessing, models, utils)
├── models/ # Modèles sauvegardés (.pkl)
├── app/ # Application Streamlit
│ ├── pages/ # 6 pages d'application
│ ├── assets/ # CSS et ressources
│ └── utils/ # Utilitaires
└── reports/figures/ # Graphiques générés
""")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>TP Deep Learning - Intelligence Artificielle 2 | DIPES 2 - 4ème année (S2) | Avril 2026</p>
    <p>Tous les modèles sont entraînés et prêts à être utilisés</p>
</div>
""", unsafe_allow_html=True)