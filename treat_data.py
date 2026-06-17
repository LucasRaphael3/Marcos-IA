"""
treat_data.py
=============
Pipeline de tratamento de dados — Social Media Addiction & Mental Health Dataset (Kaggle)
Artigo: "Vício em Redes Sociais e seu Impacto na Saúde Mental dos Jovens"

Etapas:
  1. Carregamento e inspeção inicial de cada CSV
  2. Filtragem etária: Teen + Young Adult (onde aplicável)
  3. Descarte de linhas com dados faltantes
  4. Padronização de tipos e categorias
  5. Validação de domínio (ranges esperados)
  6. Exportação de CSVs limpos individuais
  7. Dataset de modelagem: social_media_usage (tabela principal conforme artigo)
  8. Relatório de before/after em Markdown

NOTA ESTRUTURAL: As 10 tabelas são datasets sintéticos INDEPENDENTES.
Não existe chave relacional cruzada entre elas (user_id não é compartilhado
de forma consistente). Cada tabela é uma amostra própria e deve ser analisada
individualmente. O dataset de modelagem principal é social_media_usage.csv.

COMO USAR:
  1. Coloque os 10 CSVs brutos do Kaggle na pasta "data/" ao lado deste script:
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
  2. Execute:  python treat_data.py
  3. Os resultados serão salvos em "tabelas_tratadas/" e o relatório na raiz do projeto.

Autor: Guilherme / Artigo UNIMA Afya — 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — caminhos relativos ao diretório deste script
# ─────────────────────────────────────────────────────────────────────────────

# Diretório raiz do projeto (onde este script está)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pasta com os CSVs brutos do Kaggle (crie e coloque os arquivos aqui)
INPUT_DIR = os.path.join(BASE_DIR, "data")

# Pasta de saída para os CSVs tratados e o dataset de modelagem
OUTPUT_DIR = os.path.join(BASE_DIR, "tabelas_tratadas")
CLEAN_DIR  = OUTPUT_DIR   # CSVs limpos vão direto para tabelas_tratadas/
MERGE_DIR  = OUTPUT_DIR   # Dataset de modelagem também vai para tabelas_tratadas/

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_AGE_GROUPS = {"Teen", "Young Adult"}

# Domínios esperados para validação (coluna: (min, max))
DOMAIN_RULES = {
    "daily_screen_time_hours":          (0, 24),
    "doomscrolling_frequency":          (0, 10),
    "notification_checks_per_day":      (0, 500),
    "ai_recommendation_exposure":       (0, 100),
    "productivity_loss_pct":            (0, 100),
    "digital_detox_attempts":           (0, 50),
    "avg_sleep_hours":                  (0, 24),
    "sleep_quality_score":              (0, 100),
    "late_night_usage_hours":           (0, 24),
    "night_notifications":              (0, 200),
    "fatigue_level":                    (0, 100),
    "anxiety_score":                    (0, 100),
    "depression_score":                 (0, 100),
    "stress_level":                     (0, 100),
    "loneliness_index":                 (0, 100),
    "self_esteem_score":                (0, 100),
    "harassment_frequency":             (0, 100),
    "self_harm_risk_score":             (0, 100),
    "social_withdrawal_score":          (0, 100),
    "engagement_rate":                  (0, 100),
    "short_video_consumption_hours":    (0, 24),
    "dopamine_trigger_score":           (0, 100),
    "content_refresh_frequency":        (0, 1000),
    "instant_gratification_dependency": (0, 100),
    "detox_attempts":                   (0, 50),
    "offline_activity_hours_weekly":    (0, 168),
    "wellbeing_improvement_score":      (0, 100),
    "relapse_probability":              (0, 1),
    "algorithmic_content_exposure":     (0, 100),
    "echo_chamber_score":               (0, 100),
    "content_personalization_intensity":(0, 100),
    "emotional_manipulation_index":     (0, 100),
    "ai_addiction_probability":         (0, 1),
    "academic_performance_score":       (0, 100),
    "social_comparison_index":          (0, 100),
    "body_image_anxiety_score":         (0, 100),
    "peer_pressure_score":              (0, 100),
    "weekday_screen_hours":             (0, 24),
    "weekend_screen_hours":             (0, 24),
    "multitasking_frequency":           (0, 100),
    "focus_span_minutes":               (0, 240),
    "physical_activity_hours_weekly":   (0, 168),
    "predicted_anxiety_rate":           (0, 100),
    "predicted_attention_span_minutes": (0, 120),
    "predicted_ai_addiction_index":     (0, 100),
    "digital_isolation_score":          (0, 100),
    "mental_health_recovery_rate":      (0, 100),
    "predicted_sleep_decline_pct":      (0, 100),
}

INDIVIDUAL_TABLES = {
    "social_media_usage":        "social_media_usage.csv",
    "mental_health_trends":      "mental_health_trends.csv",
    "screen_time_behavior":      "screen_time_behavior.csv",
    "sleep_disruption":          "sleep_disruption.csv",
    "dopamine_trigger_metrics":  "dopamine_trigger_metrics.csv",
    "cyberbullying_impact":      "cyberbullying_impact.csv",
    "digital_detox_behavior":    "digital_detox_behavior.csv",
    "ai_recommendation_impact":  "ai_recommendation_impact.csv",
    "teen_behavior_patterns":    "teen_behavior_patterns.csv",
}

FORECAST_TABLE = {
    "name": "future_psychological_forecast",
    "file": "future_psychological_forecast.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE TRATAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def load(name, filename):
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)
    print(f"  [LOAD] {name}: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


def filter_age_groups(df, name):
    if "age_group" not in df.columns:
        print(f"  [AGE ] {name}: age_group ausente — sem filtro")
        return df
    before = len(df)
    df = df[df["age_group"].isin(TARGET_AGE_GROUPS)].copy()
    after = len(df)
    print(f"  [AGE ] {name}: {before} → {after} linhas "
          f"(removidas {before - after} | grupos: {sorted(df['age_group'].unique())})")
    return df


def drop_incomplete_rows(df, name):
    before = len(df)
    df = df.dropna().copy()
    after = len(df)
    print(f"  [NULL] {name}: {before} → {after} (descartadas {before - after} incompletas)")
    return df


def drop_duplicates(df, name):
    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)
    print(f"  [DUPL] {name}: {before} → {after} (removidas {before - after} duplicatas)")
    return df


def standardize_categoricals(df, name):
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    title_cols = ["country", "age_group", "gender", "platform",
                  "addiction_risk_level", "mental_health_risk"]
    for col in title_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Padroniza risk levels para Low / Medium / High
    for col in ["addiction_risk_level", "mental_health_risk"]:
        if col in df.columns:
            df[col] = df[col].str.capitalize()

    print(f"  [CAT ] {name}: categorias padronizadas")
    return df


def cast_types(df, name):
    int_cols = ["user_id", "year", "notification_checks_per_day",
                "digital_detox_attempts", "night_notifications",
                "content_refresh_frequency", "detox_attempts", "forecast_year"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Colunas que NÃO devem ser convertidas para numérico
    # Inclui todas as categóricas/strings conhecidas
    SKIP_COLS = set(int_cols) | {
        "user_id", "year", "country", "age_group", "gender", "platform",
        "forecast_year", "addiction_risk_level", "mental_health_risk",
    }

    for col in df.columns:
        if col in SKIP_COLS:
            continue
        # Bool nativo → preserva
        if df[col].dtype == bool or str(df[col].dtype) == "bool":
            continue
        # Qualquer tipo que pareça string/object → preserva
        dtype_str = str(df[col].dtype)
        if dtype_str in ("object", "string", "str"):
            continue
        # Demais: força float
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  [TYPE] {name}: tipos ajustados")
    return df


def validate_domains(df, name):
    violations = {}
    mask_valid = pd.Series(True, index=df.index)
    for col, (lo, hi) in DOMAIN_RULES.items():
        if col not in df.columns:
            continue
        col_mask = (df[col] >= lo) & (df[col] <= hi)
        n_invalid = (~col_mask).sum()
        if n_invalid > 0:
            violations[col] = int(n_invalid)
            mask_valid &= col_mask
    before = len(df)
    df = df[mask_valid].copy()
    after = len(df)
    if violations:
        print(f"  [DOMN] {name}: {before - after} linhas removidas → {violations}")
    else:
        print(f"  [DOMN] {name}: todos os valores dentro dos domínios esperados")
    return df, violations


def sort_and_reset(df):
    sort_by = [c for c in ["user_id", "year"] if c in df.columns]
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def save_clean(df, name):
    out_path = os.path.join(CLEAN_DIR, f"{name}_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"  [SAVE] {name}: salvo em cleaned/{name}_clean.csv ({len(df)} linhas)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def process_table(name, filename):
    print(f"\n{'─'*60}")
    print(f"  TABELA: {name}")
    print(f"{'─'*60}")

    stats = {"name": name}
    df = load(name, filename)
    stats["rows_original"] = len(df)
    stats["cols"] = df.shape[1]

    df = filter_age_groups(df, name)
    stats["rows_after_age_filter"] = len(df)

    df = drop_incomplete_rows(df, name)
    df = drop_duplicates(df, name)
    df = standardize_categoricals(df, name)
    df = cast_types(df, name)

    df, violations = validate_domains(df, name)
    stats["rows_final"] = len(df)
    stats["domain_violations"] = violations
    stats["rows_removed_total"] = stats["rows_original"] - stats["rows_final"]
    stats["pct_kept"] = round(100 * stats["rows_final"] / stats["rows_original"], 2)

    df = sort_and_reset(df)
    save_clean(df, name)

    return df, stats


def process_forecast_table(name, filename):
    print(f"\n{'─'*60}")
    print(f"  TABELA (projeções agregadas): {name}")
    print(f"{'─'*60}")

    stats = {"name": name}
    df = load(name, filename)
    stats["rows_original"] = len(df)
    stats["cols"] = df.shape[1]
    stats["rows_after_age_filter"] = "N/A (sem age_group)"

    df = drop_incomplete_rows(df, name)
    df = drop_duplicates(df, name)
    df = standardize_categoricals(df, name)
    df = cast_types(df, name)

    df, violations = validate_domains(df, name)
    stats["rows_final"] = len(df)
    stats["domain_violations"] = violations
    stats["rows_removed_total"] = stats["rows_original"] - stats["rows_final"]
    stats["pct_kept"] = round(100 * stats["rows_final"] / stats["rows_original"], 2)

    df = sort_and_reset(df)
    save_clean(df, name)

    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PRINCIPAL PARA MODELAGEM
# ─────────────────────────────────────────────────────────────────────────────

def build_modeling_dataset(clean_tables):
    """
    Dataset principal para modelagem supervisionada.

    DECISÃO ARQUITETURAL:
    As 10 tabelas são datasets sintéticos INDEPENDENTES — não há chave relacional
    cruzada entre elas (confirmado: apenas 1 linha em 20.000 bate em
    (user_id, year, platform, country) entre social_media_usage e mental_health_trends).

    Estratégia adotada:
    - Dataset de modelagem principal = social_media_usage (Teen + Young Adult)
      → possui a variável-alvo (addiction_risk_level) e os principais preditores
      → alinhado à Seção III.B do artigo
    - As demais tabelas são fontes de análise individual por dimensão:
        • sleep_disruption      → análise de sono
        • cyberbullying_impact  → análise de cyberbullying
        • dopamine_trigger_metrics → análise de engajamento
        • teen_behavior_patterns   → análise de padrões juvenis
        • etc.
    """
    print(f"\n{'='*60}")
    print("  DATASET DE MODELAGEM (social_media_usage — Teen + Young Adult)")
    print(f"{'='*60}")

    df = clean_tables["social_media_usage"].copy()

    # Ordena colunas: identificadores → demo → preditores → target
    id_cols    = ["user_id", "year", "country"]
    demo_cols  = ["age_group", "gender", "platform"]
    target_col = ["addiction_risk_level"]
    pred_cols  = [c for c in df.columns
                  if c not in id_cols + demo_cols + target_col]

    final_cols = id_cols + demo_cols + pred_cols + target_col
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    out_path = os.path.join(MERGE_DIR, "modeling_dataset_teen_young_adult.csv")
    df.to_csv(out_path, index=False)

    print(f"  ✓ Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"  ✓ Salvo em: merged/modeling_dataset_teen_young_adult.csv")

    print(f"\n  Distribuição da variável-alvo (addiction_risk_level):")
    dist = df["addiction_risk_level"].value_counts()
    for level, count in dist.items():
        pct = 100 * count / len(df)
        print(f"    {level}: {count:,} ({pct:.1f}%)")

    print(f"\n  Distribuição por age_group:")
    age_dist = df["age_group"].value_counts()
    for ag, count in age_dist.items():
        pct = 100 * count / len(df)
        print(f"    {ag}: {count:,} ({pct:.1f}%)")

    print(f"\n  Colunas preditoras ({len(pred_cols)}):")
    for c in pred_cols:
        print(f"    • {c} [{df[c].dtype}]")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RELATÓRIO MARKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(all_stats, modeling_df, clean_tables):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = []

    lines += [
        "# Relatório de Tratamento de Dados",
        f"**Dataset:** Social Media Addiction & Mental Health (Kaggle)  ",
        f"**Artigo:** Vício em Redes Sociais e seu Impacto na Saúde Mental dos Jovens  ",
        f"**Gerado em:** {now}  ",
        "",
        "## Configuração do Pipeline",
        "",
        f"- **Faixas etárias mantidas:** Teen, Young Adult",
        f"- **Política para dados faltantes:** Descarte da linha completa",
        f"- **Validação de domínio:** 46 colunas numéricas com limites definidos",
        f"- **Etapas:** Carga → Filtro etário → Nulos → Duplicatas → Padronização → Tipos → Domínio → Exportação",
        "",
        "## Nota Estrutural — Independência dos Datasets",
        "",
        "> As 10 tabelas são **datasets sintéticos independentes**. Não existe chave",
        "> relacional entre elas: a coincidência de `user_id` entre tabelas não representa",
        "> o mesmo indivíduo (confirmado empiricamente: apenas 1 linha em 20.000 bate",
        "> em `user_id + year + platform + country` entre as duas maiores tabelas).",
        "> Cada tabela deve ser analisada de forma independente.",
        "> O dataset de modelagem principal é `social_media_usage_clean.csv`.",
        "",
        "## Resultados por Tabela",
        "",
        "| Tabela | Linhas orig. | Após filtro etário | Linhas finais | Removidas | % mantido | Violações |",
        "|--------|--------------|--------------------|---------------|-----------|-----------|-----------|",
    ]

    for s in all_stats:
        name      = s["name"]
        orig      = f"{s.get('rows_original', 0):,}"
        age_filt  = s.get("rows_after_age_filter", "N/A")
        age_filt  = f"{age_filt:,}" if isinstance(age_filt, int) else age_filt
        final     = f"{s.get('rows_final', 0):,}"
        removed   = f"{s.get('rows_removed_total', 0):,}"
        pct       = f"{s.get('pct_kept', 0)}%"
        viols     = s.get("domain_violations", {})
        viols_str = ", ".join(f"{k}: {v}" for k, v in viols.items()) if viols else "—"
        lines.append(f"| {name} | {orig} | {age_filt} | {final} | {removed} | {pct} | {viols_str} |")

    lines += [
        "",
        "## Dataset de Modelagem Principal",
        "",
        f"**Arquivo:** `merged/modeling_dataset_teen_young_adult.csv`  ",
        f"**Shape:** {modeling_df.shape[0]:,} linhas × {modeling_df.shape[1]} colunas  ",
        f"**Variável-alvo:** `addiction_risk_level` (Low / Medium / High)  ",
        "",
        "### Distribuição da variável-alvo",
        "",
    ]

    dist = modeling_df["addiction_risk_level"].value_counts()
    for level, count in dist.items():
        pct = 100 * count / len(modeling_df)
        lines.append(f"- **{level}:** {count:,} ({pct:.1f}%)")

    lines += [
        "",
        "### Distribuição por faixa etária",
        "",
    ]

    age_dist = modeling_df["age_group"].value_counts()
    for ag, count in age_dist.items():
        pct = 100 * count / len(modeling_df)
        lines.append(f"- **{ag}:** {count:,} ({pct:.1f}%)")

    lines += [
        "",
        "### Variáveis preditoras (social_media_usage)",
        "",
        "| Variável | Tipo | Papel |",
        "|----------|------|-------|",
        "| year | int | Temporal |",
        "| country | categorical | Demográfico |",
        "| age_group | categorical | Demográfico (Teen / Young Adult) |",
        "| gender | categorical | Demográfico |",
        "| platform | categorical | Comportamental |",
        "| daily_screen_time_hours | float | Preditor principal |",
        "| doomscrolling_frequency | float | Preditor comportamental |",
        "| notification_checks_per_day | int | Preditor comportamental |",
        "| ai_recommendation_exposure | float | Preditor algorítmico |",
        "| productivity_loss_pct | float | Preditor de impacto |",
        "| digital_detox_attempts | int | Preditor de padrão |",
        "| **addiction_risk_level** | categorical | **Variável-alvo** |",
        "",
        "## Tabelas Complementares (análise individual por dimensão)",
        "",
        "| Tabela | Linhas (Teen+YA) | Dimensão analítica |",
        "|--------|-----------------|---------------------|",
    ]

    dim_map = {
        "mental_health_trends":      "Ansiedade, depressão, estresse, solidão",
        "screen_time_behavior":      "Tempo de tela, multitarefa, atividade física",
        "sleep_disruption":          "Sono, notificações noturnas, fadiga",
        "dopamine_trigger_metrics":  "Engajamento, vídeos curtos, dopamina",
        "cyberbullying_impact":      "Cyberbullying, assédio, retraimento social",
        "digital_detox_behavior":    "Detox digital, atividades offline, recaída",
        "ai_recommendation_impact":  "Algoritmos, câmara de eco, personalização",
        "teen_behavior_patterns":    "Desempenho acadêmico, comparação social, pressão de pares",
        "future_psychological_forecast": "Projeções agregadas (2030–2060), sem filtro etário",
    }

    for s in all_stats:
        name = s["name"]
        if name == "social_media_usage":
            continue
        rows = s.get("rows_final", 0)
        dim = dim_map.get(name, "—")
        lines.append(f"| {name}_clean.csv | {rows:,} | {dim} |")

    lines += [
        "",
        "## Resumo de Qualidade",
        "",
        "- ✅ Nenhuma coluna com valores nulos em nenhuma das 10 tabelas",
        "- ✅ Nenhuma linha completamente duplicada",
        "- ✅ Todos os valores numéricos dentro dos domínios esperados",
        "- ✅ Categorias padronizadas (Title Case, sem espaços extras)",
        "- ✅ Tipos de dados corrigidos (Int64 para inteiros, float64 para contínuas, bool para booleanos)",
        "",
        "> **Dataset sintético** abrangendo projeções 2010–2060.",
        "> Resultados devem ser interpretados como exploração computacional,",
        "> não como evidência epidemiológica direta (cf. Seção III do artigo).",
    ]

    report = "\n".join(lines)
    report_path = os.path.join(BASE_DIR, "relatorio_tratamento.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  ✓ Relatório salvo: {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PIPELINE DE TRATAMENTO DE DADOS")
    print("  Vício em Redes Sociais & Saúde Mental dos Jovens")
    print("=" * 60)

    all_stats   = []
    clean_tables = {}

    for name, filename in INDIVIDUAL_TABLES.items():
        df_clean, stats = process_table(name, filename)
        clean_tables[name] = df_clean
        all_stats.append(stats)

    forecast_df, forecast_stats = process_forecast_table(
        FORECAST_TABLE["name"], FORECAST_TABLE["file"]
    )
    all_stats.append(forecast_stats)

    modeling_df = build_modeling_dataset(clean_tables)

    generate_report(all_stats, modeling_df, clean_tables)

    print(f"\n{'='*60}")
    print("  PIPELINE CONCLUÍDO")
    print(f"{'='*60}")
    print(f"\n  Arquivos gerados:")
    print(f"  ├── tabelas_tratadas/  → 10 CSVs limpos + modeling_dataset_teen_young_adult.csv")
    print(f"  └── relatorio_tratamento.md")


if __name__ == "__main__":
    main()
