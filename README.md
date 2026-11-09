# Projeto-Final-Reconhecimento-de-Padroes
Trabalho final da cadeira de Reconhecimento de Padroes 2026.2

# 📊 Predição Conforme para Modelos Lineares Generalizados na Previsão da Nota do ENEM

> **Disciplina:** Reconhecimento de Padrões  
> **Autores:** Mateus Oliveira · Yago Pontes  
> **Dados:** Microdados ENEM 2024 — INEP  

---

## 📌 Visão Geral

Este projeto implementa um pipeline completo de **Modelos Lineares Generalizados (GLM)** com **Predição Conforme via Split Conformal** (*Split Conformal Prediction*) para a previsão das nota do Exame Nacional do Ensino Médio (ENEM) 2024.

A proposta central é construir um GLM **totalmente automatizado** — desde a identificação da distribuição da variável resposta e estimação de parâmetros até a calibração dos intervalos de predição — e quantificar a incerteza das previsões com garantias estatísticas formais providas pela predição conforme.

---

## 🏗️ Arquitetura do Pipeline

```
Entrada: X (covariáveis), y (nota pro area do conhecimento)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  1. Verificação da Condição de Carleman            │
│     └─ k momentos finitos (verificação de         │
│        regularidade para identificabilidade       │
│        da distribuição via método dos momentos)   │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  2. Análise da Variável Resposta                   │
│     ├─ Suporte (contínuo / discreto / binário)    │
│     ├─ Estatísticas descritivas                   │
│     └─ Seleção da distribuição candidata:         │
│           Normal · Gamma · Poisson ·              │
│           Binomial · Bernoulli                    │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  3. Estimação de Parâmetros                        │
│     ├─ MoM  (Método dos Momentos)                 │
│     └─ MLE  (Máxima Verossimilhança)              │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  4. Teste de Aderência (Goodness-of-Fit)           │
│     ├─ Kolmogorov–Smirnov  → distribuições        │
│     │   contínuas (Normal, Gamma)                 │
│     └─ Qui-Quadrado        → distribuições        │
│         discretas (Poisson, Binomial, Bernoulli)  │
│                                                   │
│     Critério: p-valor                             │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  5. Escolha da Função de Ligação Canônica          │
│     ├─ Normal    → Identidade:  g(μ) = μ          │
│     ├─ Gamma     → Inversa:     g(μ) = 1/μ        │
│     ├─ Poisson   → Logarítmica: g(μ) = ln(μ)      │
│     └─ Binomial / Bernoulli → Logit:              │
│         g(μ) = ln(μ / (1 − μ))                   │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  6. Treinamento — IRLS                             │
│     ├─ Iteratively Reweighted Least Squares       │
│     └─ Função de perda: Deviance da família       │
│         correspondente                            │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  7. Calibração — Split Conformal Prediction        │
│     ├─ Conjunto de calibração extraído do treino  │
│     ├─ Cálculo dos nonconformity scores           │
│     └─ Construção dos intervalos de predição      │
│         com cobertura garantida (1 − α)           │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  8. Avaliação — Conjunto de Teste                  │
│     PICP · ACE · PINAW · MPIW                     │
└───────────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Repositório

```
.
├── data/
│   ├── raw/                   # Microdados brutos ENEM 2024 (INEP)
│   └── processed/             # Dados pré-processados
│
├── notebooks/
│   ├── 01_eda.ipynb           # Análise Exploratória (boxplots, histogramas, barras)
│   ├── 02_preprocessing.ipynb # Limpeza, nulos e outliers
│   └── 03_experiments.ipynb   # Experimentos e avaliação do modelo
│
├── src/
│   ├── glm/
│   │   ├── carleman.py        # Verificação da Condição de Carleman
│   │   ├── distribution.py    # Seleção e teste de distribuição (KS / χ²)
│   │   ├── estimation.py      # MoM e MLE
│   │   ├── link_functions.py  # Identidade, Log, Logit, Inversa
│   │   └── irls.py            # Algoritmo IRLS + Deviance
│   │
│   ├── conformal/
│   │   └── split_cp.py        # Split Conformal Prediction
│   │
│   └── metrics/
│       └── coverage.py        # PICP, ACE, PINAW, MPIW
│
├── tableau/                   # Workbook e extratos para o painel
│
├── requirements.txt
└── README.md
```

---

## 📐 Fundamentos Matemáticos

### Condição de Carleman

Dado um conjunto de momentos $\{m_k\}_{k=1}^{K}$, verificamos a condição truncada:

$$\sum_{k=1}^{K} m_{2k}^{-1/(2k)} = \infty$$

A condição é verificada para $K$ momentos finitos. Na prática, assumimos que ela se satisfaz, garantindo a identificabilidade única da distribuição pelo método dos momentos.

### Famílias Exponenciais e Funções de Ligação Canônicas

| Distribuição | Suporte | Parâmetros estimados | Ligação canônica | Deviance |
|:---:|:---:|:---:|:---:|:---:|
| Normal | $\mathbb{R}$ | $\mu,\, \sigma^2$ | Identidade | $\sum (y_i - \hat\mu_i)^2$ |
| Gamma | $\mathbb{R}_{>0}$ | $\alpha,\, \beta$ | Inversa | $2\sum\left[\frac{y_i - \hat\mu_i}{\hat\mu_i} - \ln\frac{y_i}{\hat\mu_i}\right]$ |
| Poisson | $\mathbb{N}_0$ | $\lambda$ | Log | $2\sum\left[y_i \ln\frac{y_i}{\hat\mu_i} - (y_i - \hat\mu_i)\right]$ |
| Binomial | $\{0,\ldots,n\}$ | $n,\, p$ | Logit | $2\sum\left[y_i \ln\frac{y_i}{\hat\mu_i} + (n_i - y_i)\ln\frac{n_i - y_i}{n_i - \hat\mu_i}\right]$ |
| Bernoulli | $\{0,1\}$ | $p$ | Logit | Caso especial Binomial ($n=1$) |

### Split Conformal Prediction

Dado um nível de cobertura $1-\alpha$:

1. Particiona-se o conjunto de treino em $\mathcal{D}_{\text{train}}$ e $\mathcal{D}_{\text{cal}}$
2. Treina-se o GLM em $\mathcal{D}_{\text{train}}$
3. Calculam-se os *nonconformity scores* em $\mathcal{D}_{\text{cal}}$:
$$s_i = |y_i - \hat{y}_i|$$
4. Define-se o quantil empírico $\hat{q} = \text{Quantil}\!\left(\{s_i\},\, \frac{\lceil(|\mathcal{D}_{\text{cal}}|+1)(1-\alpha)\rceil}{|\mathcal{D}_{\text{cal}}|}\right)$
5. O intervalo de predição para um novo $x_{\text{new}}$ é:
$$\hat{C}(x_{\text{new}}) = \left[\hat{y}_{\text{new}} - \hat{q},\; \hat{y}_{\text{new}} + \hat{q}\right]$$

Com garantia marginal de cobertura:
$$\mathbb{P}\!\left(y_{\text{new}} \in \hat{C}(x_{\text{new}})\right) \geq 1 - \alpha$$

---

## 📊 Métricas de Avaliação

| Sigla | Nome | Descrição |
|:---:|:---|:---|
| **PICP** | Prediction Interval Coverage Probability | Fração de observações do teste cobertas pelo intervalo |
| **ACE** | Average Coverage Error | Diferença entre cobertura empírica e nominal $(1-\alpha)$ |
| **PINAW** | Prediction Interval Normalized Average Width | Largura média normalizada pelo range da variável |
| **MPIW** | Mean Prediction Interval Width | Largura média absoluta dos intervalos |

---

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/<usuario>/glm-conformal-enem.git
cd glm-conformal-enem

# Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Dependências Principais

```
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn    # apenas para utilitários de split e métricas auxiliares
statsmodels     # referência para comparação
jupyter
```

> ⚠️ Os microdados do ENEM 2024 devem ser obtidos diretamente no portal do [INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados) e colocados em `data/raw/`.

---

## 🚀 Uso

```python
from src.glm import GLMConformal

# Instanciar o modelo
model = GLMConformal(alpha=0.1)   # cobertura alvo: 90%

# Treinar (internamente realiza toda a seleção automática)
model.fit(X_train, y_train)

# Intervalos de predição no conjunto de teste
lower, upper = model.predict_interval(X_test)

# Métricas de cobertura
metrics = model.evaluate(X_test, y_test)
print(metrics)
# {'PICP': 0.912, 'ACE': 0.012, 'PINAW': 0.187, 'MPIW': 34.5}
```

---

## 📈 Painel — Tableau

Os resultados da avaliação do modelo são exportados para um painel interativo no Tableau, contendo:

- Distribuição dos intervalos de predição vs. notas reais
- PICP por subgrupo (região geográfica, tipo de escola, renda familiar)
- Largura média dos intervalos por faixa de nota
- Comparação entre distribuições candidatas testadas

---

## 🗃️ Dados

**ENEM 2024 — Microdados INEP**

- **Variável resposta:** `NU_NOTA_MT` (nota de Matemática)
- **Pré-processamento realizado:**
  - Remoção de registros com ausência (`TP_PRESENCA_MT ≠ 1`)
  - Tratamento de valores nulos nas covariáveis selecionadas
  - Detecção e análise de outliers via boxplots e IQR
  - Visualizações: histogramas, boxplots por categoria, gráficos de barras

---

## 📚 Referências

1. **Vidoni, P.** (2003). Prediction and Calibration in Generalized Linear Models. *Annals of the Institute of Statistical Mathematics*, 55(1), 169–185.

2. **Dustin, D. & Clarke, B.** (2024). Post-Model-Selection Prediction Intervals for Generalized Linear Models. *Sankhyā A: The Indian Journal of Statistics*. https://doi.org/10.1007/s13171-024-00349-7

3. **Oliveira, R. I., Orenstein, P., Ramos, T. & Romano, J. V.** (2024). Split Conformal Prediction and Non-Exchangeable Data. *Journal of Machine Learning Research*, 25, 1–38.

4. **Kim, T., Lieberman, B., Luta, G. & Peña, E. A.** (2021). Prediction intervals for Poisson-based regression models. *WIREs Computational Statistics*. https://doi.org/10.1002/wics.1568

5. **Wood, G. R.** (2005). Confidence and prediction intervals for generalised linear accident models. *Accident Analysis and Prevention*, 37(2), 267–273.

6. **Pinheiro, H. P., Rodrigues-Motta, M. & Franco, G.** (2015). Modelling performance of students with generalized linear mixed models. *Proceedings of the 60th World Statistics Congress*, ISI.

7. **Feitosa, T. C. & Braga, A. R.** (2024). Aplicações de Aprendizagem de Máquina com dados do INEP: Uma Revisão de Literatura. *IFCE / UFC*, Fortaleza–CE.

8. **Mendes, F. B.** (2024). Investigação dos Principais Fatores que Influenciam o Desempenho dos Estudantes no ENEM no Estado do Piauí, com Ênfase na Utilização de Machine Learning para Previsão das Notas. *TCC — IFPI Campus Uruçuí*.

9. **El Aissaoui, O. et al.** (2019). A Multiple Linear Regression-Based Approach to Predict Student Performance. In *Proceedings of the International Conference on Intelligent Systems and Advanced Computing Sciences*.

10. **Sangsawang, T. & Yang, L.** (2024). Predicting Student Achievement Using Socioeconomic and School-Level Factors. *Manuscript — Rajamangala University of Technology Thanyaburi*.

11. **Thiele, T., Singleton, A., Pope, D. & Stanistreet, D.** (2016). Predicting students' academic performance based on school and socio-demographic characteristics. *Studies in Higher Education*, 41(8), 1424–1446.

---

## 👥 Autores

| | Nome | GitHub |
|:---:|:---|:---|
| 👤 | **Mateus Oliveira** | [@mateusoliveira](https://github.com/mateusoliveira) |
| 👤 | **Yago Pontes** | [@yagopontes](https://github.com/yagopontes) |

---

## 📄 Licença

Este projeto é de caráter acadêmico. Consulte o arquivo [`LICENSE`](LICENSE) para detalhes.
