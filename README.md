# ☄️ Classificador de Risco de Asteroides (NEOs)

🚀 **App no ar (teste agora):** https://huggingface.co/spaces/ph-tavares/neo-hazard-classifier

O catálogo da NASA tem **dezenas de milhares** de asteroides cujas órbitas os trazem
para perto da Terra (os *NEOs* — Near-Earth Objects). A maioria é inofensiva; uma
fração, por **tamanho** e **trajetória**, merece acompanhamento. Decidir rápido
**quais** merecem atenção é uma tarefa real de defesa planetária.

Este projeto é um **classificador de risco**: você informa as características de um
asteroide e ele estima se é **potencialmente perigoso (PHA)** — com a probabilidade,
a explicação de *por quê* (SHAP) e o app acima para testar no navegador.

> Entrega da disciplina **Generative AI for Engineering (GAIE)** — FIAP, Global
> Solution 2026 (Indústria Espacial, **ODS 9 — Inovação e Infraestrutura**). É um
> pipeline completo de Machine Learning supervisionado (classificação binária).

---

## 🌍 O problema, em linguagem simples

Um asteroide é marcado pela NASA como **potencialmente perigoso (PHA)** quando cumpre
**dois critérios ao mesmo tempo**:

1. **É grande o bastante** — magnitude absoluta `H ≤ 22`, o que equivale a um diâmetro
   de aproximadamente **140 metros ou mais**; e
2. **Passa perto o bastante** — a menor distância entre a órbita dele e a da Terra
   (MOID) é `≤ 0,05 UA` (≈ **7,5 milhões de km**).

*(1 UA — unidade astronômica — ≈ 150 milhões de km, a distância média Terra–Sol.)*

Ou seja: **tamanho + proximidade**. A triagem manual desse fluxo é cara; um modelo que
faz a primeira separação — e **explica** sua decisão — é um apoio concreto.

## 🛰️ Fonte dos dados

**NASA NeoWs** (Near Earth Object Web Service), endpoint `browse`
(`https://api.nasa.gov/neo/rest/v1/neo/browse`). Coleta reproduzível em
`src/collect_neos.py`; snapshot versionado em `data/neos_raw.csv`.

- Catálogo total consultado: **61.750 NEOs**. Coletamos as páginas 0–1000 (faixa que
  possui histórico de aproximações registrado), **uma linha por asteroide**.
- Dataset final: **18.447 asteroides**, dos quais **1.919 são PHA (10,4%)**.
- **Viés de seleção (declarado):** mantemos apenas objetos com `close_approach_data`
  (necessário para as features de aproximação). É **consequência inevitável do filtro
  por disponibilidade de features**, não uma escolha para inflar a minoria. Portanto o
  modelo é um **classificador de risco para NEOs com histórico de aproximação
  registrado**, **não** um estimador da prevalência de PHA no catálogo inteiro — a
  prevalência de 10,4% aqui é maior que a do catálogo completo.

### 🔑 Como obter uma chave da NASA (grátis, opcional)

A coleta só é necessária se você quiser **recriar o dataset** do zero (o snapshot já
vem versionado). Para isso:

1. Acesse **https://api.nasa.gov/** e preencha o formulário *"Generate API Key"*
   (nome e e-mail). A chave é gratuita.
2. Exporte-a antes de rodar a coleta: `export NASA_API_KEY=<sua_chave>`.

Limites medidos: a `DEMO_KEY` pública aceita ~**10 req/hora** (suficiente só para um
teste); uma chave pessoal aceita milhares por hora (a nossa mediu **2.000 req/h**),
o que torna a coleta de ~1.000 páginas viável em poucos minutos.

### 📖 Dicionário de variáveis (o que cada número significa)

| Variável | Significado | Faixa no nosso dataset |
|---|---|---|
| `absolute_magnitude_h` | Brilho intrínseco → **proxy de tamanho** (quanto **menor** o H, **maior** o objeto) | ~10,4 a 33,2 (mediana ~23) |
| `miss_distance_min_au` | **Menor** distância de aproximação já registrada (em UA) | ~0 a 0,5 UA |
| `miss_distance_mean_au` | Distância **média** das aproximações registradas (UA) | — |
| `relative_velocity_max_kms` | **Maior** velocidade relativa registrada (km/s) | ~0,6 a 66 (média ~18,5) |
| `n_close_approaches` | **Quantas** aproximações foram registradas | 1 a 502 |
| **`is_potentially_hazardous_asteroid`** | **Alvo** (1 = PHA, 0 = não) | 10,4% positivos |

## 🔬 Metodologia

1. **Coleta** (NeoWs `browse`, 1 linha por asteroide; features de aproximação
   agregadas **apenas sobre aproximações da Terra**, pois o critério de PHA é relativo
   à Terra).
2. **Decisão anti-vazamento:** o rótulo PHA é, por definição, `tamanho (H ≤ 22) E
   proximidade orbital (MOID ≤ 0,05 UA)`. **Removemos o MOID**
   (`minimum_orbit_intersection`) — seria metade exata da regra — e também o
   `is_sentry_object`. A `absolute_magnitude_h` é mantida como variável física
   legítima; ela codifica o **critério de tamanho**, e isso é declarado abertamente.
   A dimensão de proximidade vira aprendizado real via `miss_distance`, que **não** é
   igual ao MOID (é a distância de uma passagem específica, não o mínimo orbital).
3. **Pré-processamento e split:** uma única feature de tamanho (`absolute_magnitude_h`;
   os diâmetros, derivados dela pela NeoWs, são removidos por colinearidade — ver
   heatmap no notebook). Split **estratificado em 3 conjuntos** — treino 60% /
   validação 20% / teste 20%, `random_state=42` — persistidos em `data/neos_split.csv`
   (coluna `split`). `StandardScaler` ajustado **apenas no treino** (evita vazamento de
   estatísticas de escala).
4. **Modelos:** `XGBClassifier` (com `scale_pos_weight ≈ 8,62`, calculado no treino)
   e `MLPClassifier`, treinados no conjunto de treino. Treino: **11.067** /
   Validação: **3.690** / Teste: **3.690** (PHA: 1.151 / 384 / 384).
5. **Seleção e validação:** o modelo é escolhido na **validação** (recall da classe
   PHA); as métricas finais são reportadas no **teste intocado** (accuracy, precision,
   recall, F1 com foco em PHA, `classification_report`, matriz de confusão).
6. **Interpretabilidade:** SHAP (`summary_plot`) + `plot_importance` sobre o XGBoost.

## 📊 Modelos testados e resultados

**Seleção (na VALIDAÇÃO) — critério: recall da classe PHA:**
| Modelo | Accuracy | Precision (PHA) | Recall (PHA) | F1 (PHA) |
|---|---|---|---|---|
| **XGBoost** | 0,933 | 0,623 | **0,896** | 0,735 |
| MLP | **0,960** | **0,862** | 0,732 | **0,792** |

→ **XGBoost escolhido** (maior recall na validação: 0,896 vs 0,732).

**Métricas FINAIS no TESTE intocado:**
| Modelo | Accuracy | Precision (PHA) | Recall (PHA) | F1 (PHA) |
|---|---|---|---|---|
| **XGBoost (escolhido)** | 0,935 | 0,629 | **0,917** | 0,746 |
| MLP (referência) | **0,962** | **0,877** | 0,742 | **0,804** |

**Trade-off honesto.** O `scale_pos_weight` leva o XGBoost a **recall alto** com
precisão menor; o MLP é mais equilibrado e **vence em F1 e accuracy** — reportado
abertamente. Escolhemos o XGBoost pelo **recall** (razão de domínio: em defesa
planetária, **deixar de sinalizar um asteroide perigoso — falso negativo — é muito
mais caro** que um falso alarme, que apenas gera uma observação extra). No teste, o
XGBoost deixa escapar **~8%** dos perigos (recall 0,917) contra **~26%** do MLP
(0,742). É também o modelo de árvore, interpretável diretamente com SHAP.

## 🧠 Interpretação com SHAP

Importância média (|SHAP|) sobre o XGBoost (conjunto de teste):

| Variável | Importância média (\|SHAP\|) |
|---|---|
| `absolute_magnitude_h` | 5,13 |
| `miss_distance_min_au` | 1,67 |
| `relative_velocity_max_kms` | 0,47 |
| `miss_distance_mean_au` | 0,18 |
| `n_close_approaches` | 0,18 |

O **tamanho (magnitude absoluta) e a proximidade (menor distância de aproximação)
dominam** a decisão — exatamente coerente com a definição de PHA (tamanho +
proximidade). Isso é consequência direta de manter a magnitude como feature, e está
documentado como tal.

## ⚠️ Limitações (honestas)

- **Holdout único (não k-fold CV):** usamos split estratificado em 3 conjuntos
  (treino/validação/teste) — seleção na validação, avaliação final no teste intocado.
  As métricas vêm de **um único particionamento** (com seed fixa), não de validação
  cruzada k-fold.
- **Viés de seleção:** ver seção "Fonte dos dados".
- **Vazamento parcial assumido:** a magnitude codifica o critério de tamanho; mantida
  por ser uma variável física legítima e por não esvaziar a tarefa. O MOID (critério
  orbital exato) foi removido.

## ▶️ Como executar

Requer **Python ≥ 3.10** (recomendado **3.11**, mesma versão do app no ar).

```bash
# ambiente isolado (recomendado) — use python3.11 ou outro Python >= 3.10
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt

# (opcional) recriar o dataset do zero — precisa de uma chave da NASA (ver acima):
#   export NASA_API_KEY=<sua_chave>
#   python -m src.collect_neos --pages 1000

# o notebook do pipeline já vem versionado COM os outputs; para re-executá-lo
# (treina, avalia, SHAP, salva model.joblib):
python -m nbconvert --to notebook --execute --inplace notebook.ipynb

# rodar os testes:
python -m pytest -v

# rodar o app localmente:
python app.py
```

## 🌐 Aplicação no ar

**Link persistente:** https://huggingface.co/spaces/ph-tavares/neo-hazard-classifier

Interface Gradio que carrega o modelo treinado (`model.joblib`) e classifica um
asteroide a partir das suas características, devolvendo o veredito + a probabilidade.

> Ambiente **unificado**: o mesmo `requirements.txt` (gradio 5.x, Python 3.11) roda
> localmente (venv) e no HF Space, e o `model.joblib` é treinado/serializado nesse
> mesmo ambiente.

## 🗂️ Estrutura do repositório

```
neo-hazard-classifier/
├── README.md                 # este documento
├── requirements.txt          # deps de runtime (app / HF Space)
├── requirements-dev.txt      # deps de desenvolvimento (coleta, notebook, testes)
├── data/neos_raw.csv         # snapshot da coleta (reprodutível sem API)
├── data/neos_split.csv       # mesmos dados + coluna 'split' (train/val/test)
├── src/
│   ├── neo_features.py       # lógica pura de features (testada)
│   └── collect_neos.py       # coleta NeoWs (censo + coleta paginada)
├── notebook.ipynb            # pipeline completo (EDA, treino, comparação, SHAP) — versionado com outputs
├── app.py                    # app Gradio (carrega model.joblib; não re-treina)
├── model.joblib              # modelo final + scaler + features
└── tests/                    # testes (pytest)
```

## ♻️ Reprodutibilidade

`random_state=42` em todo o pipeline; dependências versionadas; `data/neos_raw.csv`
versionado; coleta documentada em `src/collect_neos.py`. A chave da NASA nunca é
versionada (fica em `.env`, no `.gitignore`).
