# Dashboard ML — Classificador de Flores Iris

App interativo que combina exploração de dados, filtros dinâmicos e predições em tempo real com Random Forest, hospedado no Streamlit Cloud.

**[Acesse o app ao vivo](https://app-app-ddme4qe2sttvkwrwvlc8zu.streamlit.app)**

> O Streamlit Cloud coloca apps gratuitos em modo de espera após inatividade. Se aparecer uma tela pedindo para despertar o app, clique em **Yes, get this app back up!** e aguarde 30 segundos.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luccasnn/app-streamlit/blob/main/app_streamlit.ipynb)

## O que faz

App com sidebar de filtros onde o usuário seleciona espécies e os gráficos atualizam em tempo real. Na seção de predição, sliders controlam as medidas da flor e o modelo retorna a espécie com probabilidades por classe.

`@st.cache_resource` garante que o modelo seja treinado uma única vez e reutilizado em todas as interações — sem isso retreinaria a cada clique do usuário.

## Funcionalidades

- Filtro por espécie na sidebar com atualização em tempo real
- Métricas de total de flores e espécies selecionadas
- Histograma e scatter plot interativos com Plotly
- Sliders para predição interativa com resultado e probabilidades

## Tecnologias

- Streamlit — framework do app
- Plotly — gráficos interativos
- scikit-learn — modelo Random Forest
- pandas e NumPy — manipulação dos dados

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Resultado

![App Streamlit](app_streamlit.png)
