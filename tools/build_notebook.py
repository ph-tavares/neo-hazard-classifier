"""Deterministically build notebook.ipynb from explicit cell sources.

Run:  python tools/build_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
"""
import nbformat as nbf

CELLS = [
    ("markdown", """# Classificador de Risco de NEOs (PHA)

Pipeline de classificação binária supervisionada que prevê se um asteroide é
**potencialmente perigoso (PHA)**, usando apenas técnicas vistas na disciplina
(MLP, XGBoost, StandardScaler, SHAP). Fonte: **NASA NeoWs** (`browse`).

> **Nota anti-vazamento:** o rótulo PHA é definido por tamanho (H <= 22) **e**
> proximidade orbital (MOID <= 0,05 UA). Removemos o **MOID** das features; a
> magnitude absoluta (que codifica o critério de tamanho) é mantida como
> variável física legítima e isso é declarado abertamente.
>
> **Limitação declarada:** sem validação cruzada (não ensinada no fluxo manual
> da disciplina) — as métricas vêm de um único split estratificado.
"""),
    ("code", """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier, plot_importance

from src.neo_features import FEATURE_COLUMNS, TARGET_COLUMN

RANDOM_STATE = 42
"""),
    ("markdown", "## 1. Dados e EDA"),
    ("code", """df = pd.read_csv("data/neos_raw.csv")
print("shape:", df.shape)
df.head()
"""),
    ("code", """# Distribuição das classes (prevalência MEDIDA — não representa o catálogo inteiro;
# este é um conjunto de NEOs com histórico de aproximação registrado).
counts = df[TARGET_COLUMN].value_counts().sort_index()
prevalence = df[TARGET_COLUMN].mean()
print(counts)
print(f"Prevalência de PHA no conjunto: {prevalence:.1%}")
counts.plot(kind="bar")
plt.title("Distribuição das classes (0 = não-PHA, 1 = PHA)")
plt.xlabel("classe"); plt.ylabel("contagem"); plt.tight_layout(); plt.show()
"""),
    ("markdown", """### Correlação e redundância de tamanho

A NeoWs deriva `estimated_diameter` de `absolute_magnitude_h` (via albedo), então
os campos de tamanho são colineares por construção. O heatmap confirma isso e
justifica manter **uma única** feature de tamanho (`absolute_magnitude_h`).
"""),
    ("code", """size_and_features = [
    "absolute_magnitude_h", "estimated_diameter_min_km", "estimated_diameter_max_km",
    "miss_distance_min_au", "miss_distance_mean_au",
    "relative_velocity_max_kms", "n_close_approaches",
]
corr = df[size_and_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlação entre variáveis candidatas")
plt.tight_layout(); plt.show()
"""),
    ("markdown", """## 2. Pré-processamento e engenharia de atributos

- Mantemos `FEATURE_COLUMNS` (uma feature de tamanho + features de aproximação);
  os diâmetros saem por redundância (heatmap acima).
- Split **estratificado** (boa prática com classe desbalanceada).
- `StandardScaler` ajustado **apenas no treino** (evita vazamento das estatísticas
  de escala — boa prática acima do material).
"""),
    ("code", """X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# Escalamos e mantemos DataFrames nomeados (necessário para SHAP/plot_importance
# legíveis e para o XGBoost não reclamar de nomes de features).
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index
)  # fit só no treino
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index
)
print("treino:", X_train.shape, "| teste:", X_test.shape)
print("PHA no treino:", int(y_train.sum()), "| PHA no teste:", int(y_test.sum()))
"""),
    ("markdown", """## 3. Modelos: XGBoost e MLP

Duas técnicas distintas, ambas ensinadas. O XGBoost usa `scale_pos_weight` para o
desbalanceamento (Aula 06). O `MLPClassifier` do sklearn não tem equivalente —
essa assimetria é uma **hipótese** para um possível melhor desempenho do XGBoost
na classe minoritária, a confirmar pelos números.
"""),
    ("code", """n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / n_pos
print("scale_pos_weight =", round(scale_pos_weight, 3))

xgb = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    eval_metric="logloss",
)
xgb.fit(X_train_scaled, y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu",
    max_iter=500, random_state=RANDOM_STATE,
)
mlp.fit(X_train_scaled, y_train)
print("modelos treinados")
"""),
    ("markdown", "## 4. Validação e comparação\\n\\nFoco em **recall/F1 da classe PHA** (a accuracy é enganosa com desbalanceamento)."),
    ("code", """def metrics_row(name, model):
    pred = model.predict(X_test_scaled)
    return {
        "modelo": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_PHA": precision_score(y_test, pred, zero_division=0),
        "recall_PHA": recall_score(y_test, pred, zero_division=0),
        "f1_PHA": f1_score(y_test, pred, zero_division=0),
    }

comparison = pd.DataFrame([metrics_row("XGBoost", xgb), metrics_row("MLP", mlp)])
comparison
"""),
    ("code", """for name, model in [("XGBoost", xgb), ("MLP", mlp)]:
    pred = model.predict(X_test_scaled)
    print(f"===== {name} =====")
    print(classification_report(y_test, pred, digits=3, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, model) in zip(axes, [("XGBoost", xgb), ("MLP", mlp)]):
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de Confusão - {name}")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
"""),
    ("markdown", "## 5. Interpretabilidade com SHAP (sobre o XGBoost)\\n\\nO SHAP ensinado (Aula 06) é para modelos de árvore; por isso a interpretabilidade é feita sobre o XGBoost."),
    ("code", """# SHAP no MESMO espaço em que o modelo foi treinado (escalado), com nomes.
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, show=True)
"""),
    ("code", """plot_importance(xgb)
plt.tight_layout(); plt.show()
"""),
    ("markdown", """**Interpretação:** espera-se que `absolute_magnitude_h` (tamanho) e as variáveis
de proximidade (`miss_distance_*`) dominem a decisão — coerente com a definição de
PHA (tamanho + proximidade). Isso é consequência direta de manter a magnitude como
feature, e está documentado.
"""),
    ("markdown", "## 6. Escolha e persistência do modelo final\\n\\nEscolha pelo maior **F1 da classe PHA** medido. A interpretabilidade (SHAP) é feita no XGBoost independentemente."),
    ("code", """scores = {"XGBoost": (xgb, comparison.loc[0, "f1_PHA"]),
          "MLP": (mlp, comparison.loc[1, "f1_PHA"])}
best_name = max(scores, key=lambda k: scores[k][1])
best_model = scores[best_name][0]
print("modelo final (maior F1_PHA):", best_name)

joblib.dump(
    {"model": best_model, "scaler": scaler, "features": FEATURE_COLUMNS,
     "best_name": best_name},
    "model.joblib",
)
print("salvo em model.joblib")
"""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    with open("notebook.ipynb", "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("wrote notebook.ipynb with", len(nb.cells), "cells")


if __name__ == "__main__":
    main()
