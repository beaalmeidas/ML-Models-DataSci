import hdbscan
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# CARREGANDO MODELOS PARA CONSUMIR
kmeans_model = joblib.load("results/modelo_dados_bairros_hdbscan.pkl")
scaler = joblib.load("results/scaler.pkl")
nomes_bairros = joblib.load("results/nomes_bairros.pkl")

# CARREGANDO DADOS
dataset = pd.read_csv("datasets/bairros_joao_pessoa_clusterizacao.csv")
x = dataset.drop(columns=["nome_bairro"])
x_scaled = scaler.transform(x)

# Para HDBSCAN, usamos labels_ em vez de predict
labels = kmeans_model.labels_
dataset["Cluster"] = labels

# INTERFACE ------------------------------------------------------------------
st.title("Clusterização de Bairros – João Pessoa")

# MÉTRICA
# Filtra labels que não são -1 (ruído) para o silhouette
if len(set(labels)) > 1 and -1 in labels:
    sil_score = silhouette_score(x_scaled[labels != -1], labels[labels != -1])
else:
    sil_score = silhouette_score(x_scaled, labels)

st.metric("Silhouette Score", round(sil_score, 2))

# TABELA: Nomes dos bairros e o cluster ao qual pertencem
st.subheader("Bairros e seus clusters")  
st.dataframe(dataset[["nome_bairro", "Cluster"]].sort_values("Cluster"))

# GRÁFICO: Contagem de bairros por cluster (gráfico de barras)
st.subheader("Distribuição por Cluster")
st.bar_chart(dataset["Cluster"].value_counts().sort_index())

# GRÁFICO: PCA com scatter plot
st.subheader("Visualização em 2D dos Clusters (PCA)")
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(x_scaled)

fig, ax = plt.subplots()
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette="Set2", s=80, ax=ax)
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.title("Clusters de bairros - PCA")
st.pyplot(fig)

# GRÁFICO: Média dos indicadores por cluster
st.subheader("Média dos Indicadores por Cluster")
cluster_means = dataset.groupby("Cluster").mean(numeric_only=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(cluster_means.T, cmap="viridis", annot=True, fmt=".2f", ax=ax2)
plt.title("Indicadores médios por cluster (heatmap)")
st.pyplot(fig2)

