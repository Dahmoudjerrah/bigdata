
import streamlit as st
import plotly.express as px
from analyse_bourse import analyse_bourse
from prediction_models import train_models
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Récupération des données
df, df_summary = analyse_bourse()

# Entraînement des modèles et récupération des prédictions
y_test, lr_pred = train_models()

st.set_page_config(page_title='Analyse Boursière', layout='wide')

st.title("Analyse des Performances Boursières")

# Sélection du symbole boursier
symbols = df['symbol'].unique()
selected_symbol = st.sidebar.selectbox('Sélectionnez un symbole boursier', symbols)

# Filtrage des données pour le symbole sélectionné
df_symbol = df[df['symbol'] == selected_symbol]

# Évolution du cours de clôture
st.header(f"Évolution du Cours de Clôture pour {selected_symbol}")
fig = px.line(df_symbol, x="Date", y="Close", title=f"Cours de Clôture de {selected_symbol}")
st.plotly_chart(fig)

# Résumé des performances
st.header("Résumé des Performances")
st.dataframe(df_summary)

# Comparaison des prédictions avec les valeurs réelles
st.header("Prédictions vs Valeurs Réelles (Régression Linéaire)")
fig = px.scatter(x=y_test, y=lr_pred, labels={"x": "Prix Réel", "y": "Prix Prédit"})
fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
              line=dict(color="Red", width=2, dash="dash"))
st.plotly_chart(fig)

# Affichage des métriques d'évaluation
mse = mean_squared_error(y_test, lr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, lr_pred)

st.header("Métriques d'Évaluation du Modèle de Régression Linéaire")
col1, col2 = st.columns(2)
col1.metric("MSE", round(mse, 2))
col2.metric("RMSE", round(rmse, 2))
st.metric("R2", round(r2, 2))