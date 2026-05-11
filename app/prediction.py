# app/prediction.py
"""Page de déploiement du modèle - Réponse aux questions du TP"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Deploiement Modele - TP Deep Learning",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Deploiement du Modele de Prediction Bancaire")
st.markdown("---")

st.markdown("""
### Objectif du deploiement

Ce deploiement repond aux questions du TP :

1. **Enregistrer le meilleur modele** sous format .pkl avec le nom telemarketing.pkl
2. **Deployer un modele optimal** construit avec les 10 variables independantes les plus importantes
""")

# Chargement du modele
@st.cache_resource
def load_deployment_assets():
    """Charge le modele et les assets pour le deploiement"""
    
    model = None
    top_10_features = None
    scaler = None
    
    # Chemins possibles
    model_paths = [
        'models/telemarketing.pkl',
        '../models/telemarketing.pkl'
    ]
    
    top10_paths = [
        'models/top_10_features.pkl',
        '../models/top_10_features.pkl'
    ]
    
    scaler_paths = [
        'models/scalers/scaler.pkl',
        '../models/scalers/scaler.pkl'
    ]
    
    # Charger le modele
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            st.success(f"Modele charge: {path}")
            break
    
    # Charger les top 10 features
    for path in top10_paths:
        if os.path.exists(path):
            top_10_features = joblib.load(path)
            st.success(f"Top 10 features chargees: {path}")
            break
    
    # Charger le scaler
    for path in scaler_paths:
        if os.path.exists(path):
            scaler = joblib.load(path)
            break
    
    return model, top_10_features, scaler

model, top_10_features, scaler = load_deployment_assets()

# Affichage des 10 variables
st.markdown("---")
st.markdown("## Les 10 variables independantes les plus importantes")

if top_10_features is not None:
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px;">
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, feature in enumerate(top_10_features, 1):
        col_idx = 0 if i <= 5 else 1
        with cols[col_idx]:
            st.markdown(f"**{i:2d}. {feature}**")
    
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Fichier top_10_features.pkl non trouve.")

# Formulaire de prediction
st.markdown("---")
st.markdown("## Interface de prediction")
st.markdown("(Basee sur les 10 variables les plus importantes)")

if model is not None:
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Variables economiques")
        
        euribor3m = st.slider(
            "euribor3m - Taux Euribor 3 mois",
            min_value=0.0, max_value=6.0, value=2.0, step=0.1
        )
        
        nr_employed = st.slider(
            "nr.employed - Nombre d'employes",
            min_value=4900.0, max_value=5300.0, value=5100.0, step=10.0
        )
        
        emp_var_rate = st.slider(
            "emp.var.rate - Taux de variation de l'emploi",
            min_value=-5.0, max_value=5.0, value=0.0, step=0.1
        )
        
        cons_price_idx = st.slider(
            "cons.price.idx - Indice des prix",
            min_value=90.0, max_value=100.0, value=93.0, step=0.1
        )
        
        cons_conf_idx = st.slider(
            "cons.conf.idx - Indice de confiance",
            min_value=-55.0, max_value=-25.0, value=-40.0, step=1.0
        )
    
    with col2:
        st.markdown("#### Variables client")
        
        age = st.number_input(
            "age - Age du client",
            min_value=18, max_value=100, value=35
        )
        
        duration = st.slider(
            "duration - Duree de l'appel (secondes)",
            min_value=0, max_value=5000, value=200
        )
        
        campaign = st.number_input(
            "campaign - Nombre de contacts",
            min_value=1, max_value=50, value=1
        )
        
        pdays = st.number_input(
            "pdays - Jours depuis dernier contact",
            min_value=-1, max_value=999, value=999
        )
        
        previous = st.number_input(
            "previous - Contacts precedents",
            min_value=0, max_value=50, value=0
        )
    
    # Bouton de prediction
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("PREDIRE LA SOUSCRIPTION", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyse en cours avec les 10 variables importantes..."):
            # Construction des donnees
            input_data = pd.DataFrame([{
                'euribor3m': euribor3m,
                'nr.employed': nr_employed,
                'emp.var.rate': emp_var_rate,
                'cons.price.idx': cons_price_idx,
                'cons.conf.idx': cons_conf_idx,
                'age': age,
                'duration': duration,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous
            }])
            
            # Normalisation
            if scaler is not None:
                try:
                    input_scaled = scaler.transform(input_data)
                except:
                    input_scaled = input_data.values
            else:
                input_scaled = input_data.values
            
            # Prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Affichage
            st.markdown("---")
            st.markdown("## Resultat de la prediction")
            
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            
            with col_res2:
                if prediction == 1:
                    st.success(f"""
                    ### SOUSCRIPTION PREDITE : OUI
                    
                    **Probabilite : {probability:.1%}**
                    
                    Ce client a une forte probabilite de souscrire.
                    """)
                else:
                    st.warning(f"""
                    ### SOUSCRIPTION PREDITE : NON
                    
                    **Probabilite : {probability:.1%}**
                    
                    Ce client a une faible probabilite de souscrire.
                    """)
                
                st.progress(probability)

else:
    st.error("""
    **Modele non trouve !**
    
    Veuillez d'abord executer le script run_project.py pour :
    1. Entrainer les modeles
    2. Selectionner les 10 meilleures variables
    3. Sauvegarder le modele telemarketing.pkl
    
    Commande: python run_project.py
    """)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: gray;">
    TP Deep Learning - Deploiement du modele avec les 10 variables les plus importantes
</p>
""", unsafe_allow_html=True)