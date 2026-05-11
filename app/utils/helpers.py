# app/utils/helpers.py
import streamlit as st
import plotly.graph_objects as go

def render_plotly_with_zoom(fig, key, use_container_width=True, modal_width="large"):
    """
    Affiche un graphique Plotly avec un bouton "Agrandir" qui ouvre une fenêtre modale
    contenant le même graphique en plus grand.
    """
    # Affichage normal
    st.plotly_chart(fig, use_container_width=use_container_width, key=f"plot_{key}")
    
    # Bouton pour agrandir
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔍 Agrandir", key=f"zoom_btn_{key}", use_container_width=True):
            # Ouvre une fenêtre modale avec le graphique
            with st.dialog("📊 Graphique agrandi", width=modal_width):
                st.plotly_chart(fig, use_container_width=True)
                
                # Option : ajouter un bouton de fermeture
                if st.button("✖ Fermer", use_container_width=True):
                    st.rerun()