"""
ml_pipeline.py
==============
Pipeline de Machine Learning — Social Media Addiction & Mental Health Dataset (Kaggle)
Artigo: "Vício em Redes Sociais e seu Impacto na Saúde Mental dos Jovens"

PRÉ-REQUISITO:
  Execute treat_data.py primeiro. Este script lê o arquivo gerado por ele:
    tabelas_tratadas/modeling_dataset_teen_young_adult.csv

DEPENDÊNCIAS:
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap tabulate

USO:
  python ml_pipeline.py

SAÍDAS (em tabelas_tratadas/):
  fig1_confusion_matrices.png  — Matrizes de confusão (3 modelos)
  fig2_model_comparison.png    — Comparação de métricas + CV
  fig3_feature_importance.png  — Importância das variáveis (RF e XGBoost)
  fig4_shap_summary.png        — SHAP Summary Plot (XGBoost, classe High)
  fig5_f1_per_class.png        — F1-Score por classe
  metrics_summary.csv          — Tabela de métricas exportável
  metrics_summary.md           — Relatório Markdown com detalhamento por classe

Modelos: Random Forest · Hist Gradient Boosting · XGBoost
Tarefa:  Classificação multiclasse (Low / Medium / High addiction risk)

Autor: Guilherme / Artigo UNIMA Afya — 2025
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display — funciona em qualquer máquina
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — caminhos relativos ao diretório deste script
# ─────────────────────────────────────────────────────────────────────────────

# Diretório raiz do projeto (onde este script está)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho para o dataset gerado pelo treat_data.py
DATA_PATH = os.path.join(BASE_DIR, "tabelas_tratadas", "modeling_dataset_teen_young_adult.csv")

# Diretório de saída para figuras e métricas (mesma pasta dos dados tratados)
OUTPUT_DIR = os.path.join(BASE_DIR, "tabelas_tratadas")

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80% treino / 20% teste
CV_FOLDS     = 3      # validação cruzada estratificada

TARGET      = "addiction_risk_level"
LABEL_ORDER = ["Low", "Medium", "High"]  # ordem das classes (Low=0, Medium=1, High=2)

# Colunas preditoras
CAT_COLS = ["country", "age_group", "gender", "platform"]
NUM_COLS = [
    "year",
    "daily_screen_time_hours",
    "doomscrolling_frequency",
    "notification_checks_per_day",
    "ai_recommendation_exposure",
    "productivity_loss_pct",
    "digital_detox_attempts",
]

# Paletas visuais
MODEL_COLORS = {
    "Random Forest":          "#1565C0",
    "Hist Gradient Boosting": "#6A1B9A",
    "XGBoost":                "#E65100",
}
CLASS_COLORS = {
    "Low":    "#4CAF50",
    "Medium": "#FFC107",
    "High":   "#F44336",
}

# Estilo global dos gráficos
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    print("=" * 62)
    print("  1. CARREGAMENTO E PRÉ-PROCESSAMENTO")
    print("=" * 62)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATA_PATH}\n"
            "Execute treat_data.py antes de rodar este script."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    X     = df.drop(columns=[TARGET])
    y     = df[TARGET]

    # Encode ordinal do target: Low=0, Medium=1, High=2
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_enc = le.transform(y)

    dist = dict(zip(le.classes_, np.bincount(y_enc)))
    print(f"  Distribuição target: {dist}")

    # Pré-processador: StandardScaler para numéricas, OHE para categóricas
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)

    ohe_names     = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS)
    feature_names = NUM_COLS + list(ohe_names)

    print(f"  Split: treino={len(X_train):,}  teste={len(X_test):,}  (estratificado {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})")
    print(f"  Features após OHE: {len(feature_names)}")

    return X_train_prep, X_test_prep, y_train, y_test, feature_names, le


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINIÇÃO DOS MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",    # compensa desbalanceamento de classes
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,  # HistGB: nativo, ~10× mais rápido que GBM clássico
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. TREINAMENTO, CROSS-VALIDATION E AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test, le):
    print("\n" + "=" * 62)
    print("  2. TREINAMENTO E AVALIAÇÃO")
    print("=" * 62)

    results   = {}
    cms       = {}
    cv_scores = {}

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  ── {name}")

        # Cross-validation
        cv = cross_validate(
            model, X_train, y_train,
            cv=skf,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            n_jobs=-1,
        )
        cv_scores[name] = cv
        print(f"  CV accuracy  ({CV_FOLDS}-fold): {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}")
        print(f"  CV F1-macro  ({CV_FOLDS}-fold): {cv['test_f1_macro'].mean():.4f} ± {cv['test_f1_macro'].std():.4f}")

        # Treino no conjunto completo de treino e avaliação no teste
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model":       model,
            "y_pred":      y_pred,
            "accuracy":    accuracy_score(y_test, y_pred),
            "f1_macro":    f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision":   precision_score(y_test, y_pred, average="macro"),
            "recall":      recall_score(y_test, y_pred, average="macro"),
            "report":      classification_report(
                               y_test, y_pred,
                               target_names=le.classes_,
                               output_dict=True,
                           ),
        }
        cms[name] = confusion_matrix(y_test, y_pred)

        print(f"  Accuracy:     {results[name]['accuracy']:.4f}")
        print(f"  F1-Macro:     {results[name]['f1_macro']:.4f}")
        print(f"  F1-Weighted:  {results[name]['f1_weighted']:.4f}")
        print(f"  Precision:    {results[name]['precision']:.4f}")
        print(f"  Recall:       {results[name]['recall']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    return results, cms, cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(cms, le, results):
    """Fig 1 — Matrizes de confusão normalizadas (3 modelos lado a lado)."""
    print("  [FIG1] Matrizes de confusão...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Matrizes de Confusão — Classificação de Risco de Vício (Teen + Young Adult)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, (name, cm) in zip(axes, cms.items()):
        acc     = results[name]["accuracy"]
        f1      = results[name]["f1_macro"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax, linewidths=0.5, linecolor="#e0e0e0",
            cbar=False, annot_kws={"size": 10},
        )
        # Contagens absolutas abaixo de cada percentual
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                ax.text(j + 0.5, i + 0.72, f"(n={cm[i,j]:,})",
                        ha="center", va="center", fontsize=7.5, color="#555")

        ax.set_title(f"{name}\nAcc={acc:.3f}  F1-macro={f1:.3f}", pad=10, fontsize=12)
        ax.set_xlabel("Predito", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        ax.set_xticklabels(le.classes_, rotation=0)
        ax.set_yticklabels(le.classes_, rotation=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_confusion_matrices.png")
    plt.savefig(path)
    plt.close()
    print(f"         → {path}")


def plot_model_comparison(results, cv_scores):
    """Fig 2 — Comparação de métricas no teste + CV com barras de erro."""
    print("  [FIG2] Comparação de modelos...")

    metrics_map = {
        "Accuracy":    [r["accuracy"]     for r in results.values()],
        "F1-Macro":    [r["f1_macro"]     for r in results.values()],
        "F1-Weighted": [r["f1_weighted"]  for r in results.values()],
        "Precision":   [r["precision"]    for r in results.values()],
        "Recall":      [r["recall"]       for r in results.values()],
    }
    mnames = list(results.keys())
    x      = np.arange(len(metrics_map))
    w      = 0.22
    offsets = [-w, 0, w]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Comparação de Desempenho dos Modelos", fontsize=14, fontweight="bold")

    # Subplot esquerdo: métricas no conjunto de teste
    ax = axes[0]
    for i, (mname, color) in enumerate(MODEL_COLORS.items()):
        vals = [metrics_map[m][i] for m in metrics_map]
        bars = ax.bar(x + offsets[i], vals, w, label=mname, color=color, alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_map.keys()), fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Métricas no Conjunto de Teste ({int(TEST_SIZE*100)}%)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Subplot direito: CV accuracy com barras de erro
    ax2    = axes[1]
    cv_means = [cv_scores[m]["test_accuracy"].mean() for m in mnames]
    cv_stds  = [cv_scores[m]["test_accuracy"].std()  for m in mnames]
    colors   = [MODEL_COLORS[m] for m in mnames]

    bars = ax2.bar(mnames, cv_means, color=colors, alpha=0.85,
                   yerr=cv_stds, capsize=6, error_kw={"linewidth": 1.5}, zorder=3)
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + 0.007,
                 f"{mean:.4f} ± {std:.4f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Accuracy (CV)", fontsize=11)
    ax2.set_title(f"Validação Cruzada — {CV_FOLDS}-Fold Estratificado", fontsize=12)
    ax2.set_xticklabels(mnames, fontsize=9)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_model_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"         → {path}")


def plot_feature_importance(results, feature_names):
    """Fig 3 — Feature importance intrínseca (RF e XGBoost, Top 15)."""
    print("  [FIG3] Feature importance...")

    TOP_N    = 15
    fi_models = {
        "Random Forest": results["Random Forest"]["model"].feature_importances_,
        "XGBoost":       results["XGBoost"]["model"].feature_importances_,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Importância das Variáveis Preditoras (Top 15)",
                 fontsize=14, fontweight="bold")

    for ax, (mname, importances) in zip(axes, fi_models.items()):
        color = MODEL_COLORS[mname]
        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=True)
            .tail(TOP_N)
        )
        # Nomes legíveis para o gráfico
        fi_df["label"] = (
            fi_df["feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
            .str.replace("age_group_", "", regex=False)
            .str.replace("platform_",  "", regex=False)
            .str.replace("gender_",    "", regex=False)
            .str.replace("country_",   "", regex=False)
            .str.replace("_", " ",     regex=False)
        )

        bars = ax.barh(fi_df["label"], fi_df["importance"],
                       color=color, alpha=0.82, height=0.65)
        for bar, val in zip(bars, fi_df["importance"]):
            ax.text(bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8)

        ax.set_title(f"{mname}", fontsize=12)
        ax.set_xlabel("Importância (impurity / gain)", fontsize=10)
        ax.set_xlim(0, fi_df["importance"].max() * 1.20)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_feature_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"         → {path}")


def plot_shap(results, X_test, feature_names, le):
    """Fig 4 — SHAP Summary Plot (XGBoost, classe High Risk, 300 amostras)."""
    print("  [FIG4] SHAP summary plot...")

    xgb_model = results["XGBoost"]["model"]
    explainer  = shap.TreeExplainer(xgb_model)

    # Amostragem para não travar em máquinas mais lentas
    n_sample = min(300, X_test.shape[0])
    rng      = np.random.default_rng(RANDOM_STATE)
    idx      = rng.choice(X_test.shape[0], size=n_sample, replace=False)
    X_sample = X_test[idx]

    # shap_values shape: (n_samples, n_features, n_classes)
    shap_vals = explainer.shap_values(X_sample)
    class_idx = list(le.classes_).index("High")

    # Labels legíveis
    labels = [
        n.replace("cat__", "").replace("num__", "")
         .replace("age_group_", "").replace("platform_", "")
         .replace("gender_", "").replace("country_", "")
         .replace("_", " ")
        for n in feature_names
    ]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals[:, :, class_idx],
        X_sample,
        feature_names=labels,
        max_display=15,
        show=False,
        plot_type="dot",
    )
    plt.title(
        "SHAP Summary Plot — Classe 'High Risk' (XGBoost)\n"
        "Impacto de cada variável na predição de alto risco de vício",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_shap_summary.png")
    plt.savefig(path)
    plt.close()
    print(f"         → {path}")


def plot_f1_per_class(results, le):
    """Fig 5 — F1-Score por classe de risco para cada modelo."""
    print("  [FIG5] F1 por classe...")

    classes  = list(le.classes_)
    x        = np.arange(len(classes))
    w        = 0.25
    offsets  = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (mname, res) in enumerate(results.items()):
        f1s  = [res["report"][c]["f1-score"] for c in classes]
        bars = ax.bar(x + offsets[i], f1s, w - 0.02,
                      label=mname, color=MODEL_COLORS[mname],
                      alpha=0.85, zorder=3)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("F1-Score por Classe de Risco — Comparação entre Modelos",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Fundo suave por classe
    for xi, cls in enumerate(classes):
        ax.axvspan(xi - 0.43, xi + 0.43, alpha=0.04,
                   color=CLASS_COLORS[cls], zorder=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_f1_per_class.png")
    plt.savefig(path)
    plt.close()
    print(f"         → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPORTAÇÃO DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(results, cv_scores, le):
    print("\n" + "=" * 62)
    print("  4. EXPORTANDO MÉTRICAS")
    print("=" * 62)

    rows = []
    for name, res in results.items():
        cv_a = cv_scores[name]["test_accuracy"]
        cv_f = cv_scores[name]["test_f1_macro"]
        rows.append({
            "Modelo":               name,
            "Accuracy (test)":      round(res["accuracy"],    4),
            "F1-Macro (test)":      round(res["f1_macro"],    4),
            "F1-Weighted (test)":   round(res["f1_weighted"], 4),
            "Precision (test)":     round(res["precision"],   4),
            "Recall (test)":        round(res["recall"],      4),
            f"CV Accuracy ({CV_FOLDS}-fold)":
                f"{cv_a.mean():.4f} ± {cv_a.std():.4f}",
            f"CV F1-Macro ({CV_FOLDS}-fold)":
                f"{cv_f.mean():.4f} ± {cv_f.std():.4f}",
        })

    df_metrics = pd.DataFrame(rows)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"  metrics_summary.csv → {csv_path}")

    # Markdown
    md_path = os.path.join(OUTPUT_DIR, "metrics_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Resumo de Métricas — ML Pipeline\n\n")
        f.write("**Dataset:** `modeling_dataset_teen_young_adult.csv`  \n")
        f.write("**Recorte:** Teen + Young Adult (20.001 registros)  \n")
        f.write("**Target:** `addiction_risk_level` (Low / Medium / High)  \n")
        f.write(f"**Split:** {int((1-TEST_SIZE)*100)}% treino / {int(TEST_SIZE*100)}% teste (estratificado)  \n")
        f.write(f"**CV:** {CV_FOLDS}-fold estratificado  \n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n\n## Detalhamento por Classe\n\n")
        for name, res in results.items():
            f.write(f"### {name}\n\n")
            f.write("| Classe | Precision | Recall | F1-Score | Support |\n")
            f.write("|--------|-----------|--------|----------|---------|\n")
            for cls in LABEL_ORDER:
                r = res["report"][cls]
                f.write(
                    f"| {cls} | {r['precision']:.3f} | {r['recall']:.3f} "
                    f"| {r['f1-score']:.3f} | {int(r['support'])} |\n"
                )
            f.write("\n")

    print(f"  metrics_summary.md  → {md_path}")
    return df_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("  ML PIPELINE — VÍCIO EM REDES SOCIAIS & SAÚDE MENTAL")
    print("=" * 62)

    # 1. Pré-processamento
    X_train, X_test, y_train, y_test, feature_names, le = load_and_preprocess()

    # 2. Treino + avaliação
    models                     = build_models()
    results, cms, cv_scores    = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, le
    )

    # 3. Visualizações
    print("\n" + "=" * 62)
    print("  3. VISUALIZAÇÕES")
    print("=" * 62)
    plot_confusion_matrices(cms, le, results)
    plot_model_comparison(results, cv_scores)
    plot_feature_importance(results, feature_names)
    plot_shap(results, X_test, feature_names, le)
    plot_f1_per_class(results, le)

    # 4. Métricas
    save_metrics(results, cv_scores, le)

    # 5. Ranking final
    print("\n" + "=" * 62)
    print("  CONCLUÍDO")
    print("=" * 62)
    print("\n  Ranking final (F1-Macro no conjunto de teste):")
    ranking = sorted(results.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    for i, (name, res) in enumerate(ranking, 1):
        print(f"  {i}. {name:<28}  F1-Macro={res['f1_macro']:.4f}  "
              f"Accuracy={res['accuracy']:.4f}")

    print(f"\n  Saídas em: tabelas_tratadas/")


if __name__ == "__main__":
    main()
