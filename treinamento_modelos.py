"""
Engenharia de ML — Social Media Addiction Risk
===============================================
Responsável: Time de ML Engineering

Pipeline de treinamento independente.
Entrada : social_media_usage_tratado.csv  (gerado por tratamento_dados.py)
Saída   : modelos/*.pkl  (para o time de QA/Interpretabilidade avaliar)

Etapas:
  1. Carregamento e encoding do alvo
  2. Separação treino/teste estratificada
  3. Instanciação dos modelos
  4. Treinamento e exportação
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("[AVISO] xgboost não instalado. Rode: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_OK = True
except ImportError:
    LIGHTGBM_OK = False
    print("[AVISO] lightgbm não instalado. Rode: pip install lightgbm")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
INPUT_FILE    = "social_media_usage_tratado.csv"
MODELS_DIR    = "modelos"
VARIAVEL_ALVO = "addiction_risk_level"
SEED          = 42
TEST_SIZE     = 0.20

# Low=0  Medium=1  High=2  (ordem crescente de risco)
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E ENCODING DO ALVO
# ─────────────────────────────────────────────
def carregar_dados(caminho: str):
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Execute tratamento_dados.py primeiro."
        )

    df = pd.read_csv(caminho)

    print("=" * 55)
    print("1. CARREGAMENTO")
    print("=" * 55)
    print(f"   Shape: {df.shape}")

    # Encoding do alvo
    df[VARIAVEL_ALVO] = df[VARIAVEL_ALVO].map(LABEL_MAP)

    dist = df[VARIAVEL_ALVO].value_counts().sort_index()
    nomes = {v: k for k, v in LABEL_MAP.items()}
    print("   Distribuição do alvo:")
    for idx, n in dist.items():
        print(f"     {nomes[idx]:8s} ({idx}): {n:,}  ({n/len(df)*100:.1f}%)")
    print()

    y = df[VARIAVEL_ALVO].values
    X = df.drop(columns=[VARIAVEL_ALVO]).values
    feature_names = df.drop(columns=[VARIAVEL_ALVO]).columns.tolist()

    return X, y, feature_names


# ─────────────────────────────────────────────
# 2. SEPARAÇÃO TREINO / TESTE ESTRATIFICADA
# ─────────────────────────────────────────────
def splittar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,        # garante proporção das 3 classes nos dois splits
    )

    print("=" * 55)
    print("2. SPLIT TREINO / TESTE")
    print("=" * 55)
    print(f"   Treino : {X_train.shape[0]:,} registros")
    print(f"   Teste  : {X_test.shape[0]:,} registros")
    print(f"   Features: {X_train.shape[1]}")
    print()

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. INSTANCIAÇÃO DOS MODELOS
# ─────────────────────────────────────────────
def criar_modelos() -> dict:
    modelos = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=SEED,
            multi_class="multinomial",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=SEED,
        ),
    }

    if XGBOOST_OK:
        modelos["xgboost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=SEED,
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,
        )

    if LIGHTGBM_OK:
        modelos["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=-1,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    print("=" * 55)
    print("3. MODELOS INSTANCIADOS")
    print("=" * 55)
    for nome in modelos:
        print(f"   ✔ {nome}")
    print()

    return modelos


# ─────────────────────────────────────────────
# 4. TREINAMENTO E EXPORTAÇÃO
# ─────────────────────────────────────────────
def treinar_e_exportar(modelos: dict, X_train, y_train, feature_names: list):
    print("=" * 55)
    print("4. TREINAMENTO E EXPORTAÇÃO")
    print("=" * 55)

    for nome, modelo in modelos.items():
        print(f"\n   [{nome}] Treinando...", end=" ", flush=True)
        modelo.fit(X_train, y_train)
        print("✔")

        # Salva o modelo em .pkl
        caminho_pkl = os.path.join(MODELS_DIR, f"{nome}.pkl")
        with open(caminho_pkl, "wb") as f:
            pickle.dump(modelo, f)
        print(f"   Salvo em: {caminho_pkl}")

    # Salva os nomes das features para o time de QA usar no SHAP
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"\n   Feature names salvas em: {features_path}")
    print()


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  PIPELINE DE TREINAMENTO — ML ENGINEERING")
    print("=" * 55 + "\n")

    X, y, feature_names = carregar_dados(INPUT_FILE)
    X_train, X_test, y_train, y_test = splittar(X, y)
    modelos = criar_modelos()
    treinar_e_exportar(modelos, X_train, y_train, feature_names)

    # Salva os splits para o time de QA reproduzir as avaliações
    splits_path = os.path.join(MODELS_DIR, "splits.pkl")
    with open(splits_path, "wb") as f:
        pickle.dump({"X_train": X_train, "X_test": X_test,
                     "y_train": y_train, "y_test": y_test}, f)
    print(f"   Splits salvos em: {splits_path}")

    print("\n✅ Treinamento concluído!")
    print(f"   Modelos disponíveis em: ./{MODELS_DIR}/\n")
    print("   Entregue ao time de QA:")
    print(f"     - {MODELS_DIR}/*.pkl")
    print(f"     - {MODELS_DIR}/feature_names.pkl")
    print(f"     - {MODELS_DIR}/splits.pkl\n")


if __name__ == "__main__":
    main()
