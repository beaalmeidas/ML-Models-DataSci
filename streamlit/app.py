import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# CARREGANDO MODELOS PARA CONSUMIR
gmm_model = joblib.load("results/modelo_gmm_bairros.pkl")
scaler = joblib.load("results/scaler.pkl")

# CARREGANDO DADOS
dataset = pd.read_csv("datasets/bairros_joao_pessoa_clusterizacao.csv")
x = dataset.drop(columns=["nome_bairro"])
x_scaled = scaler.transform(x)

# APLICANDO PCA
pca = PCA(n_components=3, random_state=42)
x_pca = pca.fit_transform(x_scaled)

# PREDIÇÃO DOS CLUSTERS
labels = gmm_model.predict(x_pca)
dataset["Cluster"] = labels

# INTERFACE -----------------------------------------------------------------------------------
st.set_page_config(page_title="Clusterização de Bairros - João Pessoa", layout="wide")

st.title("📍 Clusterização de Bairros – João Pessoa")
st.markdown("""Esse aplicativo busca agrupar os bairros da cidade de João Pessoa (PB) de acordo
            com características socioeconômicas. Isso é feito através de um modelo de 
            clusterização implementado com o algoritmo Gaussian Mixture Models (GMM).
""")

# MÉTRICA: Mostrando a nota do modelo no silhouette score --------------------------------------
with st.container():
    st.subheader("📊 Avaliação do Modelo")
    sil_score = silhouette_score(x_pca, labels)
    st.metric("Silhouette Score", round(sil_score, 3))
    st.markdown("O Silhouette Score mede a qualidade da separação entre os grupos encontrados.")

# ----------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)

# TABELA: Nomes dos bairros e o cluster ao qual pertencem
with col1:
    st.subheader("Bairros e seus clusters")  
    st.dataframe(dataset[["nome_bairro", "Cluster"]].sort_values("Cluster"))

# GRÁFICO: Contagem de bairros por cluster (gráfico de barras)
with col2:
    st.subheader("Distribuição por Cluster")
    st.bar_chart(dataset["Cluster"].value_counts().sort_index())

# ----------------------------------------------------------------------------------------------
col3, col4 = st.columns(2)

# GRÁFICO: Plotagem dos clusters com PCA
with col3:
    st.subheader("Visualização em 2D dos Clusters (PCA)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette="Set2", s=80, ax=ax)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.title("Clusters de bairros - PCA")
    st.pyplot(fig)

# GRÁFICO: Média dos indicadores por cluster
with col4:
    st.subheader("Média dos Indicadores por Cluster")
    cluster_means = dataset.groupby("Cluster").mean(numeric_only=True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_means.T, cmap="viridis", annot=True, fmt=".2f", ax=ax2)
    plt.title("Indicadores médios por cluster (heatmap)")
    st.pyplot(fig2)

# GRÁFICO: Média de uma variável específica por cluster --------------------------------------
col5, col6, col7 = st.columns([1, 2, 1])

with col6:
    st.subheader("Comparar variável por cluster")
    st.markdown(
        "Selecione uma variável socioeconômica abaixo para visualizar sua média por cluster. "
        "Isso ajuda a entender o perfil médio de cada grupo formado pela clusterização."
    )

    # Select e cálculo do valor médio da variável
    variavel_escolhida = st.selectbox("Escolha uma variável:", x.columns)
    medias_por_cluster = dataset.groupby("Cluster")[variavel_escolhida].mean()

    fig3, ax3 = plt.subplots(figsize=(5, 3))  # Tamanho menor
    medias_por_cluster.plot(kind='bar', ax=ax3, color='skyblue')
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel(f"Média de {variavel_escolhida}")
    ax3.set_title(f"Média de '{variavel_escolhida}' por Cluster")
    plt.xticks(rotation=0)
    st.pyplot(fig3)
