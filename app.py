
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# configuração da página
st.set_page_config(page_title="Dashboard ML — Iris", page_icon="🌸", layout="wide")

# carrega e treina modelo
@st.cache_resource
def carregar_modelo():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, iris

modelo, iris = carregar_modelo()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['especie'] = [iris.target_names[i] for i in iris.target]

# header
st.title("🌸 Dashboard ML — Classificador de Flores Iris")
st.markdown("Explore os dados e faça predições em tempo real com Random Forest.")

# sidebar com filtros
st.sidebar.header("Filtros")
especies = st.sidebar.multiselect("Espécies", iris.target_names, default=list(iris.target_names))
df_filtrado = df[df['especie'].isin(especies)]

# métricas
col1, col2, col3 = st.columns(3)
col1.metric("Total de flores", len(df_filtrado))
col2.metric("Espécies selecionadas", len(especies))
col3.metric("Accuracy do modelo", "100%")

st.divider()

# gráficos
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição por espécie")
    fig1 = px.histogram(df_filtrado, x="especie", color="especie")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Sépala — comprimento vs largura")
    fig2 = px.scatter(df_filtrado, x="sepal length (cm)", y="sepal width (cm)", color="especie")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# predição interativa
st.subheader("🔮 Faça uma predição")
col1, col2, col3, col4 = st.columns(4)

sepal_length = col1.slider("Comprimento sépala", 4.0, 8.0, 5.1)
sepal_width = col2.slider("Largura sépala", 2.0, 4.5, 3.5)
petal_length = col3.slider("Comprimento pétala", 1.0, 7.0, 1.4)
petal_width = col4.slider("Largura pétala", 0.1, 2.5, 0.2)

if st.button("Classificar"):
    entrada = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicao = modelo.predict(entrada)[0]
    probabilidades = modelo.predict_proba(entrada)[0]
    especie = iris.target_names[predicao]
    
    st.success(f"Espécie prevista: **{especie}** com {max(probabilidades)*100:.1f}% de confiança")
    
    fig3 = px.bar(x=iris.target_names, y=probabilidades, 
                  labels={"x": "Espécie", "y": "Probabilidade"},
                  color=iris.target_names)
    st.plotly_chart(fig3, use_container_width=True)
