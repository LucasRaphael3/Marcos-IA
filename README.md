# 📱 Vício em Redes Sociais e seu Impacto na Saúde Mental dos Jovens

> Estudo de caso e pipeline de Machine Learning desenvolvido para o artigo científico da UNIMA Afya — 2025.  
> Dataset: [Social Media Addiction & Mental Health (Kaggle)](https://www.kaggle.com/)

---

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
  - [1. Preparar os dados brutos](#1-preparar-os-dados-brutos)
  - [2. Executar o tratamento de dados](#2-executar-o-tratamento-de-dados)
  - [3. Executar o pipeline de ML](#3-executar-o-pipeline-de-ml)
- [Saídas Geradas](#saídas-geradas)
- [Nota sobre os Datasets](#nota-sobre-os-datasets)

---

## Sobre o Projeto

Este repositório contém o pipeline completo de análise de dados e Machine Learning para o artigo:

**"Vício em Redes Sociais e seu Impacto na Saúde Mental dos Jovens"**

O objetivo é classificar o risco de vício em redes sociais (`Low / Medium / High`) para jovens nas faixas **Teen** e **Young Adult**, com base em variáveis comportamentais, de tempo de tela e exposição algorítmica.

**Modelos utilizados:** Random Forest · Hist Gradient Boosting · XGBoost

---

## Estrutura do Repositório

```
📦 projeto/
│
├── 📄 treat_data.py              # Pipeline de tratamento e limpeza dos dados
├── 📄 ml_pipeline.py             # Pipeline de Machine Learning (treino + avaliação)
│
├── 📁 data/                      # ← Coloque aqui os CSVs brutos do Kaggle
│   └── .gitkeep                  #   (pasta rastreada, CSVs ignorados pelo git)
│
├── 📁 tabelas_tratadas/          # Saída gerada pelo treat_data.py
│   ├── social_media_usage_clean.csv
│   ├── mental_health_trends_clean.csv
│   ├── screen_time_behavior_clean.csv
│   ├── sleep_disruption_clean.csv
│   ├── dopamine_trigger_metrics_clean.csv
│   ├── cyberbullying_impact_clean.csv
│   ├── digital_detox_behavior_clean.csv
│   ├── ai_recommendation_impact_clean.csv
│   ├── teen_behavior_patterns_clean.csv
│   ├── future_psychological_forecast_clean.csv
│   └── modeling_dataset_teen_young_adult.csv  # Dataset principal de modelagem
│
├── 📄 relatorio_tratamento.md    # Relatório de qualidade gerado pelo treat_data.py
└── 📄 .gitignore
```

---

## Pré-requisitos

- Python **3.9+**
- pip

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/LucasRaphael3/Marcos-IA.git
cd Marcos-IA

# (Recomendado) Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap tabulate
```

---

## Como Usar

### 1. Preparar os dados brutos

Faça o download dos 10 CSVs do dataset no Kaggle e coloque-os dentro da pasta `data/`:

```
data/
  social_media_usage.csv
  mental_health_trends.csv
  screen_time_behavior.csv
  sleep_disruption.csv
  dopamine_trigger_metrics.csv
  cyberbullying_impact.csv
  digital_detox_behavior.csv
  ai_recommendation_impact.csv
  teen_behavior_patterns.csv
  future_psychological_forecast.csv
```

> **Nota:** os arquivos CSV brutos são ignorados pelo git (`.gitignore`). Apenas a estrutura da pasta é versionada.

---

### 2. Executar o tratamento de dados

```bash
python treat_data.py
```

O script realiza as seguintes etapas em cada tabela:

| Etapa | Descrição |
|-------|-----------|
| **Carga** | Leitura dos CSVs brutos da pasta `data/` |
| **Filtro etário** | Mantém apenas `Teen` e `Young Adult` |
| **Nulos** | Descarta linhas com qualquer valor faltante |
| **Duplicatas** | Remove linhas completamente duplicadas |
| **Padronização** | Title Case em categóricas, strip de espaços |
| **Tipos** | Int64 para inteiros, float64 para contínuas |
| **Validação de domínio** | Verifica 46 colunas numéricas com limites definidos |
| **Exportação** | Salva CSVs limpos em `tabelas_tratadas/` |

**Saídas geradas:**
- `tabelas_tratadas/*_clean.csv` — 10 tabelas limpas individuais
- `tabelas_tratadas/modeling_dataset_teen_young_adult.csv` — dataset de modelagem
- `relatorio_tratamento.md` — relatório de qualidade com estatísticas before/after

---

### 3. Executar o pipeline de ML

> **Pré-requisito:** executar o `treat_data.py` primeiro (ou usar os CSVs já presentes em `tabelas_tratadas/`).

```bash
python ml_pipeline.py
```

O pipeline realiza:

| Etapa | Descrição |
|-------|-----------|
| **Pré-processamento** | StandardScaler (numéricas) + OneHotEncoder (categóricas) |
| **Split** | 80% treino / 20% teste, estratificado por classe |
| **Treinamento** | 3 modelos: Random Forest, Hist Gradient Boosting, XGBoost |
| **Cross-validation** | 3-fold estratificado (accuracy + F1-macro) |
| **Avaliação** | Accuracy, F1-Macro, F1-Weighted, Precision, Recall |
| **Visualizações** | 5 gráficos gerados automaticamente |
| **Exportação** | Métricas em CSV e Markdown |

---

## Saídas Geradas

Todos os arquivos de saída são salvos em `tabelas_tratadas/`:

| Arquivo | Descrição |
|---------|-----------|
| `fig1_confusion_matrices.png` | Matrizes de confusão normalizadas (3 modelos) |
| `fig2_model_comparison.png` | Comparação de métricas + CV com barras de erro |
| `fig3_feature_importance.png` | Importância das variáveis — RF e XGBoost (Top 15) |
| `fig4_shap_summary.png` | SHAP Summary Plot — XGBoost, classe High Risk |
| `fig5_f1_per_class.png` | F1-Score por classe de risco |
| `metrics_summary.csv` | Tabela de métricas exportável |
| `metrics_summary.md` | Relatório Markdown com detalhamento por classe |

---

## Nota sobre os Datasets

As 10 tabelas são **datasets sintéticos independentes** — não existe chave relacional entre elas. A coincidência de `user_id` entre tabelas não representa o mesmo indivíduo. Cada tabela deve ser analisada de forma independente.

| Tabela | Dimensão analítica |
|--------|--------------------|
| `social_media_usage` | **Tabela principal** — variável-alvo (`addiction_risk_level`) |
| `mental_health_trends` | Ansiedade, depressão, estresse, solidão |
| `screen_time_behavior` | Tempo de tela, multitarefa, atividade física |
| `sleep_disruption` | Sono, notificações noturnas, fadiga |
| `dopamine_trigger_metrics` | Engajamento, vídeos curtos, dopamina |
| `cyberbullying_impact` | Cyberbullying, assédio, retraimento social |
| `digital_detox_behavior` | Detox digital, atividades offline, recaída |
| `ai_recommendation_impact` | Algoritmos, câmara de eco, personalização |
| `teen_behavior_patterns` | Desempenho acadêmico, comparação social, pressão de pares |
| `future_psychological_forecast` | Projeções agregadas 2030–2060 (sem filtro etário) |

> **Aviso:** Os resultados devem ser interpretados como exploração computacional, não como evidência epidemiológica direta (cf. Seção III do artigo).

---

**Autor:** Guilherme · UNIMA Afya — 2025
