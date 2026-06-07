# Classificador de Risco de NEOs (PHA)

Pipeline completo de Machine Learning que classifica **objetos próximos da Terra (NEOs)** como **potencialmente perigosos (PHA)** ou não, a partir de características físicas e de aproximação. Entrega da disciplina **Generative AI for Engineering (GAIE)** — FIAP, Global Solution 2026 (Indústria Espacial, **ODS 9 — Inovação e Infraestrutura**).

> Observação: apesar do nome da disciplina ("Generative AI"), o conteúdo lecionado e exigido é **Machine Learning supervisionado clássico** (MLP, XGBoost, SHAP). Esta entrega é um classificador supervisionado, não IA generativa.

## Contexto do problema
Asteroides cujas órbitas os trazem para perto da Terra (NEOs) são monitorados continuamente por agências como NASA/JPL. Uma fração é classificada como **potencialmente perigosa (PHA)** com base em tamanho e proximidade orbital. A triagem rápida — separar o que merece acompanhamento do que é inofensivo — é uma tarefa central de defesa planetária e de *Space Situational Awareness*, um setor real da economia espacial. Este projeto entrega um classificador de risco para apoiar essa triagem, com interpretabilidade via SHAP.

## Fonte dos dados
**NASA NeoWs** (Near Earth Object Web Service), endpoint `browse`
(`https://api.nasa.gov/neo/rest/v1/neo/browse`). Coleta reproduzível em `src/collect_neos.py`; snapshot versionado em `data/neos_raw.csv`.

- Catálogo total: **61.750 NEOs**. Coletamos as páginas 0–1000 (faixa que possui histórico de aproximações registrado), uma linha por asteroide.
- Dataset final: **18.447 asteroides**, dos quais **1.919 são PHA (10,4%)**.
- **Viés de seleção (declarado):** mantemos apenas objetos com `close_approach_data` (necessário para as features de aproximação). Isso é **consequência inevitável do filtro por disponibilidade de features**, não uma escolha para inflar a minoria. Por isso o modelo é um **classificador de risco para NEOs com histórico de aproximação registrado**, **não** um estimador da prevalência de PHA no catálogo inteiro — a prevalência de 10,4% aqui é maior que a do catálogo completo.

### Dicionário de variáveis (features do modelo)
| Variável | Significado |
|---|---|
| `absolute_magnitude_h` | Magnitude absoluta (proxy de tamanho; menor H = maior) |
| `miss_distance_min_au` | Menor distância de aproximação já registrada (UA) |
| `miss_distance_mean_au` | Distância média das aproximações (UA) |
| `relative_velocity_max_kms` | Velocidade relativa máxima registrada (km/s) |
| `n_close_approaches` | Número de aproximações registradas |
| **`is_potentially_hazardous_asteroid`** | **Alvo** (1 = PHA, 0 = não) |

## Metodologia
1. **Coleta** (NeoWs `browse`, 1 linha por asteroide; features de aproximação agregadas **apenas sobre aproximações da Terra**, pois o critério de PHA é relativo à Terra).
2. **Decisão anti-vazamento:** o rótulo PHA é, por definição, `tamanho (H ≤ 22) E proximidade orbital (MOID ≤ 0,05 UA)`. **Removemos o MOID** (`minimum_orbit_intersection`) — seria metade exata da regra — e também o `is_sentry_object`. A `absolute_magnitude_h` é mantida como variável física legítima; ela codifica o **critério de tamanho**, e isso é declarado abertamente. A dimensão de proximidade vira aprendizado real via `miss_distance` (que **não** é igual ao MOID).
3. **Pré-processamento:** uma única feature de tamanho (`absolute_magnitude_h`; os diâmetros, derivados dela pela NeoWs, são removidos por colinearidade — ver heatmap no notebook). `StandardScaler` ajustado **apenas no conjunto de treino** (evita vazamento de estatísticas de escala). Split **estratificado**, `random_state=42`.
4. **Modelos:** `XGBClassifier` (com `scale_pos_weight` para o desbalanceamento) e `MLPClassifier`. Treino: 12.912 / Teste: 5.535 (PHA: 1.343 / 576). `scale_pos_weight ≈ 8,61`.
5. **Validação:** accuracy, precision, recall e F1 (foco na classe PHA), `classification_report` e matriz de confusão.
6. **Interpretabilidade:** SHAP (`summary_plot`) + `plot_importance` sobre o XGBoost.

## Modelos testados e resultados
| Modelo | Accuracy | Precision (PHA) | Recall (PHA) | F1 (PHA) |
|---|---|---|---|---|
| **XGBoost** | 0,933 | 0,619 | **0,927** | 0,743 |
| MLP | **0,962** | **0,856** | 0,766 | **0,808** |

**Trade-off honesto.** O `scale_pos_weight` leva o XGBoost a **recall alto (0,93)** com precisão menor (0,62); o MLP é mais equilibrado e **vence em F1 e accuracy**. Os dois resultados são reportados abertamente.

**Modelo final: XGBoost**, selecionado pelo **recall da classe PHA**. Razão de domínio: em triagem de defesa planetária, **deixar de sinalizar um asteroide perigoso (falso negativo) é muito mais caro** do que um falso alarme (que apenas gera observação de acompanhamento). O XGBoost deixa escapar ~7% dos perigos (recall 0,93) contra ~23% do MLP (recall 0,77). É também o modelo de árvore interpretável via o SHAP ensinado.

## Interpretação com SHAP
Importância média (|SHAP|) sobre o XGBoost:

| Variável | Importância média (\|SHAP\|) |
|---|---|
| `absolute_magnitude_h` | 5,18 |
| `miss_distance_min_au` | 1,67 |
| `relative_velocity_max_kms` | 0,39 |
| `n_close_approaches` | 0,22 |
| `miss_distance_mean_au` | 0,14 |

O **tamanho (magnitude absoluta) e a proximidade (menor distância de aproximação) dominam** a decisão — coerente com a definição de PHA (tamanho + proximidade). Isso é consequência direta de manter a magnitude como feature, e está documentado como tal.

## Limitações (honestas)
- **Sem validação cruzada:** o fluxo manual da disciplina usa um único `train_test_split`; CV manual (`cross_val_score`/`KFold`) não foi ensinada nesse fluxo. As métricas vêm de **um único split estratificado** — limitação declarada, não defeito.
- **Viés de seleção:** ver seção "Fonte dos dados".
- **Vazamento parcial assumido:** a magnitude codifica o critério de tamanho; mantida por ser uma variável física legítima e por não esvaziar a tarefa. O MOID (critério orbital exato) foi removido.

## Como executar
```bash
python -m pip install -r requirements-dev.txt

# (opcional) recriar o dataset do zero — precisa de uma chave da NASA:
#   export NASA_API_KEY=<sua_chave>
#   python -m src.collect_neos --pages 1000

# gerar e executar o notebook do pipeline (treina, avalia, SHAP, salva model.joblib):
python tools/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebook.ipynb

# rodar os testes:
python -m pytest -v

# rodar o app localmente:
python app.py
```

## Aplicação no ar
**[A PUBLICAR]** — Hugging Face Space (Gradio). O link persistente será inserido aqui após o deploy.

## Estrutura do repositório
```
neo-hazard-classifier/
├── README.md                 # este documento
├── requirements.txt          # deps de runtime (app / HF Space)
├── requirements-dev.txt      # deps de desenvolvimento (coleta, notebook, testes)
├── data/neos_raw.csv         # snapshot da coleta (reprodutível sem API)
├── src/
│   ├── neo_features.py       # lógica pura de features (testada)
│   └── collect_neos.py       # coleta NeoWs (censo + coleta paginada)
├── tools/build_notebook.py   # gera notebook.ipynb deterministicamente
├── notebook.ipynb            # pipeline completo (EDA, treino, comparação, SHAP)
├── app.py                    # app Gradio (carrega model.joblib; não re-treina)
├── model.joblib              # modelo final + scaler + features
└── tests/                    # testes (pytest)
```

## Reprodutibilidade
`random_state=42` em todo o pipeline; dependências versionadas; `data/neos_raw.csv` versionado; coleta documentada em `src/collect_neos.py`. A chave da NASA nunca é versionada (fica em `.env`, no `.gitignore`).
