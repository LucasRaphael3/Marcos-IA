# Relatório de Tratamento de Dados
**Dataset:** Social Media Addiction & Mental Health (Kaggle)  
**Artigo:** Vício em Redes Sociais e seu Impacto na Saúde Mental dos Jovens  
**Gerado em:** 17/06/2026 19:38  

## Configuração do Pipeline

- **Faixas etárias mantidas:** Teen, Young Adult
- **Política para dados faltantes:** Descarte da linha completa
- **Validação de domínio:** 46 colunas numéricas com limites definidos
- **Etapas:** Carga → Filtro etário → Nulos → Duplicatas → Padronização → Tipos → Domínio → Exportação

## Nota Estrutural — Independência dos Datasets

> As 10 tabelas são **datasets sintéticos independentes**. Não existe chave
> relacional entre elas: a coincidência de `user_id` entre tabelas não representa
> o mesmo indivíduo (confirmado empiricamente: apenas 1 linha em 20.000 bate
> em `user_id + year + platform + country` entre as duas maiores tabelas).
> Cada tabela deve ser analisada de forma independente.
> O dataset de modelagem principal é `social_media_usage_clean.csv`.

## Resultados por Tabela

| Tabela | Linhas orig. | Após filtro etário | Linhas finais | Removidas | % mantido | Violações |
|--------|--------------|--------------------|---------------|-----------|-----------|-----------|
| social_media_usage | 50,000 | 20,001 | 20,001 | 29,999 | 40.0% | — |
| mental_health_trends | 50,000 | 20,067 | 20,067 | 29,933 | 40.13% | — |
| screen_time_behavior | 50,000 | 19,998 | 19,998 | 30,002 | 40.0% | — |
| sleep_disruption | 50,000 | 19,966 | 19,966 | 30,034 | 39.93% | — |
| dopamine_trigger_metrics | 50,000 | 20,043 | 20,043 | 29,957 | 40.09% | — |
| cyberbullying_impact | 50,000 | 20,083 | 20,083 | 29,917 | 40.17% | — |
| digital_detox_behavior | 50,000 | 19,880 | 19,880 | 30,120 | 39.76% | — |
| ai_recommendation_impact | 50,000 | 19,980 | 19,980 | 30,020 | 39.96% | — |
| teen_behavior_patterns | 40,000 | 40,000 | 40,000 | 0 | 100.0% | — |
| future_psychological_forecast | 30,000 | N/A (sem age_group) | 30,000 | 0 | 100.0% | — |

## Dataset de Modelagem Principal

**Arquivo:** `merged/modeling_dataset_teen_young_adult.csv`  
**Shape:** 20,001 linhas × 13 colunas  
**Variável-alvo:** `addiction_risk_level` (Low / Medium / High)  

### Distribuição da variável-alvo

- **Medium:** 9,987 (49.9%)
- **Low:** 6,087 (30.4%)
- **High:** 3,927 (19.6%)

### Distribuição por faixa etária

- **Teen:** 10,018 (50.1%)
- **Young Adult:** 9,983 (49.9%)

### Variáveis preditoras (social_media_usage)

| Variável | Tipo | Papel |
|----------|------|-------|
| year | int | Temporal |
| country | categorical | Demográfico |
| age_group | categorical | Demográfico (Teen / Young Adult) |
| gender | categorical | Demográfico |
| platform | categorical | Comportamental |
| daily_screen_time_hours | float | Preditor principal |
| doomscrolling_frequency | float | Preditor comportamental |
| notification_checks_per_day | int | Preditor comportamental |
| ai_recommendation_exposure | float | Preditor algorítmico |
| productivity_loss_pct | float | Preditor de impacto |
| digital_detox_attempts | int | Preditor de padrão |
| **addiction_risk_level** | categorical | **Variável-alvo** |

## Tabelas Complementares (análise individual por dimensão)

| Tabela | Linhas (Teen+YA) | Dimensão analítica |
|--------|-----------------|---------------------|
| mental_health_trends_clean.csv | 20,067 | Ansiedade, depressão, estresse, solidão |
| screen_time_behavior_clean.csv | 19,998 | Tempo de tela, multitarefa, atividade física |
| sleep_disruption_clean.csv | 19,966 | Sono, notificações noturnas, fadiga |
| dopamine_trigger_metrics_clean.csv | 20,043 | Engajamento, vídeos curtos, dopamina |
| cyberbullying_impact_clean.csv | 20,083 | Cyberbullying, assédio, retraimento social |
| digital_detox_behavior_clean.csv | 19,880 | Detox digital, atividades offline, recaída |
| ai_recommendation_impact_clean.csv | 19,980 | Algoritmos, câmara de eco, personalização |
| teen_behavior_patterns_clean.csv | 40,000 | Desempenho acadêmico, comparação social, pressão de pares |
| future_psychological_forecast_clean.csv | 30,000 | Projeções agregadas (2030–2060), sem filtro etário |

## Resumo de Qualidade

- ✅ Nenhuma coluna com valores nulos em nenhuma das 10 tabelas
- ✅ Nenhuma linha completamente duplicada
- ✅ Todos os valores numéricos dentro dos domínios esperados
- ✅ Categorias padronizadas (Title Case, sem espaços extras)
- ✅ Tipos de dados corrigidos (Int64 para inteiros, float64 para contínuas, bool para booleanos)

> **Dataset sintético** abrangendo projeções 2010–2060.
> Resultados devem ser interpretados como exploração computacional,
> não como evidência epidemiológica direta (cf. Seção III do artigo).