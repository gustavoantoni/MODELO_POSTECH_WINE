# 🍷 Classificação da Qualidade de Vinhos com Machine Learning

> **Tech Challenge — Fase 2 | POSTECH DTAT**  
> Projeto de pós-graduação em Data Analytics 

---

## Visão Geral

A avaliação da qualidade de um vinho é tradicionalmente feita por especialistas por meio de análises sensoriais — um processo subjetivo, custoso e dependente de experiência humana. Este projeto desenvolve um modelo de Machine Learning capaz de prever a qualidade de vinhos tintos a partir de suas características físico-químicas, utilizando o [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) disponível publicamente no Kaggle.

O problema foi formulado como uma **classificação binária**:

| Classe | Critério |
|--------|----------|
| 🟢 Alta Qualidade | Nota ≥ 7 |
| 🔴 Baixa/Média Qualidade | Nota < 7 |

---

## Estrutura do Repositório

```
wine-quality-classification/
│
│
├── data/
│   └── WineQT.csv                  # Base de dados utilizada
│
├── notebooks/
│   └── Versao_completa.ipynb       # Notebook principal — contém toda a análise,
│                                   # visualizações, modelagem e resultados
│
├── requirements.txt                # Bibliotecas utilizadas
└── README.md                       # Descrição do projeto
```

---

## Pipeline do Projeto

### 1. Compreensão do Problema
Interpretação do contexto vitivinícola, definição da variável alvo e transformação da nota de qualidade em classificação binária (≥7 = Alta Qualidade).

### 2. Análise Exploratória de Dados (EDA)
- Distribuição de todas as variáveis físico-químicas com histogramas e KDE por classe
- Matriz de correlação com identificação de multicolinearidade
- Identificação de outliers com **três métodos combinados**: IQR, Z-Score e Isolation Forest, com índices de concordância (Cohen's Kappa e Jaccard)
- Análise do balanceamento das classes (~86% baixa/média × ~14% alta qualidade)

### 3. Pré-processamento
- Remoção da coluna `Id` e verificação de dados faltantes
- **Tratamento de outliers por consenso**: remoção apenas dos registros identificados pelos três métodos simultaneamente (~7% do dataset), preservando vinhos genuinamente atípicos
- **Feature Engineering** — 3 variáveis derivadas criadas a partir dos insights da EDA:
  - `alcohol_acid_ratio` — razão álcool / acidez volátil
  - `free_so2_ratio` — proporção SO₂ livre / SO₂ total
  - `high_alcohol` — flag binária para teor alcoólico > 11%
- Padronização com `StandardScaler` aplicado apenas no treino (sem *data leakage*)

### 4. Modelos Treinados
Quatro algoritmos foram treinados e comparados:

| Modelo | Estratégia de balanceamento |
|--------|----------------------------|
| Decision Tree | `class_weight='balanced'` |
| Random Forest | GridSearchCV + `class_weight='balanced'` |
| KNN | Melhor K via validação cruzada |
| Regressão Logística | `class_weight='balanced'` |

Todos os modelos foram também treinados com **SMOTE** (*Synthetic Minority Oversampling Technique*) para comparação direta do impacto do balanceamento.

O **Random Forest** foi otimizado via `GridSearchCV` com 5-fold cross-validation, buscando maximizar o F1-score da classe Alta Qualidade, explorando combinações de `n_estimators`, `max_depth` e `min_samples_leaf`.

### 5. Avaliação
- Métricas: Acurácia, Precisão, Recall, F1-Score e AUC-ROC
- Validação cruzada com 5 folds para todos os modelos
- Curvas ROC sobrepostas para comparação visual
- Tabela comparativa consolidada Sem SMOTE × Com SMOTE
- Teste com casos reais extraídos diretamente do dataset

### 6. Interpretação dos Resultados
- Feature Importance do Random Forest e coeficientes da Regressão Logística
- Discussão das variáveis mais relevantes e suas implicações práticas no processo produtivo

---

## Principais Resultados

O **Random Forest** (com GridSearchCV e SMOTE) apresentou o melhor desempenho geral, com maior AUC-ROC e melhor equilíbrio entre precisão e recall para a classe minoritária (Alta Qualidade).

As variáveis de maior influência identificadas pelos modelos foram:

| Variável | Direção | Interpretação |
|----------|---------|---------------|
| `alcohol` | ✅ Positiva | Principal preditor de alta qualidade |
| `alcohol_acid_ratio` | ✅ Positiva | Combinação favorável álcool/acidez |
| `sulphates` | ✅ Positiva | Agente conservante e antimicrobiano |
| `volatile acidity` | ❌ Negativa | Indica deterioração microbiológica |

> ⚠️ **Nota sobre métricas**: a acurácia global não é o indicador mais adequado para este problema dado o desbalanceamento das classes. Um modelo que classifique tudo como "baixa/média" já atingiria ~86% de acurácia sem nenhum poder preditivo real. **Recall e AUC-ROC** são as métricas prioritárias.

---

## Como Executar

### Pré-requisitos
Python 3.8+ e as bibliotecas listadas em `requirements.txt`.

```bash
# Instalar dependências
pip install -r requirements.txt

# Abrir o notebook
jupyter notebook notebooks/Versao_completa.ipynb
```

### Dados
Faça o download do dataset [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) e salve o arquivo `WineQT.csv` na pasta `data/`.

---

## Tecnologias Utilizadas

| Biblioteca | Uso |
|------------|-----|
| `pandas` / `numpy` | Manipulação e análise de dados |
| `matplotlib` / `seaborn` | Visualizações |
| `scipy` | Estatísticas (Z-Score, testes) |
| `scikit-learn` | Modelos, métricas, pré-processamento, GridSearchCV |
| `imbalanced-learn` | SMOTE para balanceamento de classes |

---


## Referências

- [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- [POSTECH DTAT — Tech Challenge Fase 2](POSTECH_-_DTAT_-_Tech_Challenge_-_Fase_2.pdf)
