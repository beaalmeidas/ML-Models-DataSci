# Modelos de Machine Learning Un.1 – Ciência de Dados

**Descrição:** Esse repositório contém 3 modelos de machine learning em Python, desenvolvidos como atividade prática na Unidade 1 da disciplina Ciência de Dados, curso de Ciência da Computação 2025.1

---

## Componentes:
- **Modelo 1:** Clusterização de bairros em João Pessoa de acordo com medidores socioeconômicos
- **Modelo 2:** Classificação Binária de tumores de câncer de mama
- **Modelo 3:** Regressão Linear de Dados de Cultivo de Milho

#### Datasets:
- **(1)** Dados socioeconômicos fictícios de bairros de João Pessoa: Projetado por um dos autores e gerado com ChatGPT.
- **(2)** Dados técnicos sobre tumores de câncer de mama: Adquirido no Kaggle → [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- **(3)** Dados fictícios de cultivo de milho para previsão de produtividade: Projetado por um dos autores e gerado com ChatGPT.

---

## Como executar

O projeto do modelo 1 (Clusterização de bairros de JP) tem deploy no Streamlit, e está disponível no link: https://bairros-jp-clusters.streamlit.app/

Para rodar o Streamlit localmente, usar o comando:
```
streamlit run streamlit/app.py
```

### Com uma IDE:
**1.** Clone o repositório  
[https://github.com/beaalmeidas/ML-models-ECMD.git](https://github.com/beaalmeidas/ML-models-ECMD.git)
```
git clone https://github.com/beaalmeidas/ML-models-ECMD.git
```

**2.** Abra a pasta do repositório em uma IDE

**3.** Instale as dependências
```
pip install -r requirements.txt
```

**4.** Escolha um notebook

**5.** Os datasets estarão presentes na pasta 'datasets'. Edite o caminho do dataset para incluir a pasta de origem. Deve ficar assim (exemplo):
```
breast_cancer_data = pd.read_csv("datasets/breast_cancer.csv", encoding="utf-8")
```

**6.** Execute cada célula de código

### Com o [Google Colab](https://colab.research.google.com):
**1.** Baixe o(s) notebook(s) desejado(s) ou o arquivo zip do repositório

**2.** Faça o upload do(s) notebook(s) para sua pasta do Colab Drive

**3.** Adicione o dataset indicado aos arquivos do projeto no Colab

**4.** Execute cada célula de código

---

## Créditos

**Autores:** Beatriz Almeida de Souza Silva, Rigel Sales de Souza
</br>
**Professor:** Ricardo Roberto de Lima