# Methods

Este arquivo descreve as especificacoes reais dos tres pipelines de vetorizacao, o calculo de similaridade e as metricas de avaliacao IR implementadas no codigo atual.

## Modelo base compartilhado

Arquivo: `core/src/infrastructure/encoders/base.py` (`SharedSbertBase`)

- Modelo: `sentence-transformers/all-MiniLM-L6-v2` (configuravel via `CLASSICAL_MODEL_NAME`)
- Dimensao de saida bruta: 384
- `normalize_embeddings=False`: normalizacao L2 e delegada a cada pipeline
- Instancia unica cacheada globalmente (`_MODEL_CACHE`)

## Pipeline 1 - Classico

Arquivo: `core/src/infrastructure/encoders/classical.py` (`ClassicalPipelineEncoder`)

```
SBERT(384) → PCA(n=64, random_state=seed) → L2 normalize → dim=64
```

- PCA ajustada sobre os embeddings brutos do corpus (fit unico na indexacao)
- Resultado armazenado em `documents.embedding_vector vector(64)`

## Pipeline 2 - Quantico-inspirado (Residual Quantum Feature Map)

Arquivo: `core/src/infrastructure/encoders/quantum.py` (`QuantumPipelineEncoder`)

```
SBERT(384)
  → PCA_base(n=64)            → base_vector_64
  → PCA_angles(n=6)           → 6 angulos
  → normalize_angles([0, π])  → angulos_norm
  → AngleEmbedding(Y) + StronglyEntanglingLayers(2 camadas, 6 qubits)
  → qml.probs(wires=[0..5])   → probs_64 (2^6 = 64)
  → sqrt(probs_64)            → quantum_vector_64  [transformacao Hellinger]
  → concat(base_64, quantum_64) → vector_128
  → PCA_final(n=64)           → L2 normalize → dim=64
```

**Detalhes do circuito** (`default.qubit`, PennyLane):
- 6 qubits (`QUANTUM_N_QUBITS=6`), 2 camadas `StronglyEntanglingLayers`
- Pesos do circuito fixos por semente (`SEED=42`), nao treinados
- Invariante: `2 ** QUANTUM_N_QUBITS == VECTOR_DIM` (enforced em `config.py`)

**Tres PCAs ajustadas sequencialmente durante indexacao**:
1. `PCA_base(64)` sobre embeddings brutos do corpus
2. `PCA_angles(6)` sobre os vetores base (normaliza angulos ao range [0, π] por min/max por componente)
3. `PCA_final(64)` sobre os vetores concatenados de 128 dimensoes

**Transformacao Hellinger**: `sqrt(abs(probs))` — mantem a geometria de distribuicao de probabilidades

Resultado armazenado em `documents.quantum_vector vector(64)`

## Pipeline 3 - Estatistico

Arquivo: `core/src/infrastructure/encoders/statistical.py` (`StatisticalPipelineEncoder`)

```
SBERT(384)
  → PCA(n=128, random_state=seed)          → base_vector_128
  → TruncatedSVD(n=64, random_state=seed) → L2 normalize → dim=64
```

- Fatoracao linear em dois estagios: PCA(128) centraliza e reduz o espaco; TruncatedSVD(64) fatoriza para a dimensao final
- `PCA_INTERMEDIATE_DIM = 128 > VECTOR_DIM = 64` — o SVD realiza reducao real de dimensionalidade, descobrindo um subespa co diferente do PCA(64) isolado
- Se `PCA_INTERMEDIATE_DIM == VECTOR_DIM`, o SVD degeneraria para uma rotacao ortogonal e a similaridade cosseno seria identica ao pipeline classico
- Ambas as transformacoes ajustadas sobre o corpus na indexacao

Logs emitidos durante encode:
```
[PIPELINE statistical] base_vector_dim=128 svd_input_dim=128 svd_output_dim=64
[NORMALIZE] vector_norm=1.0
```

Resultado armazenado em `documents.statistical_vector vector(64)`

## Normalizacao L2

Arquivo: `core/src/domain/ir.py` (`l2_normalize`, `DEFAULT_TOP_K`, `ALLOWED_TOP_K`)

**`DEFAULT_TOP_K = 25`**: valor padrao do numero de documentos recuperados. **`ALLOWED_TOP_K = (10, 25, 50, 100)`**: valores permitidos. O usuario pode escolher o valor de `top_k` no frontend; valores fora do conjunto permitido sao substituidos pelo padrao.

Para um vetor `v = [v1, v2, ..., vn]`:

- Norma L2: `||v||2 = sqrt(sum(vi^2))`
- Vetor normalizado: `v_hat = v / ||v||2`

Caso especial: se `||v||2 = 0`, retorna vetor de zeros com o mesmo tamanho.

Todos os tres pipelines aplicam `l2_normalize` como etapa final antes do armazenamento e na busca.

## Similaridade / Score de busca

Arquivo: `core/src/infrastructure/repositories/sqlalchemy_repositories.py`

- Banco ordena por `cosine_distance` (pgvector)
- Score exposto pela API: `score = 1 - cosine_distance(query_vector, doc_vector)`
- Vetores L2-normalizados → cosine_distance equivale a similaridade cosseno direta

## Modo compare (comparacao entre pipelines)

Arquivo: `core/src/application/ir_use_cases.py`

No modo `compare`, os tres pipelines sao executados em paralelo. A resposta inclui:

- `comparison.classical`, `comparison.quantum`, `comparison.statistical` — top-k de cada pipeline
- `comparison_metrics`:
  - `common_doc_ids` — intersecao dos tres top-k
  - `common_classical_quantum` — intersecao classical ∩ quantum
  - `common_classical_statistical` — intersecao classical ∩ statistical
  - `common_quantum_statistical` — intersecao quantum ∩ statistical

## Metricas de avaliacao IR

Arquivo: `core/src/infrastructure/metrics/ir_measures_adapter.py` (`IrMeasuresAdapter`)

Metricas calculadas pela biblioteca `ir_measures` (padrao da area, sem implementacao manual):

| Metrica | Descricao |
|---|---|
| `nDCG@k` | Normalized Discounted Cumulative Gain at k (default 25, configuravel para 10, 50, 100) |
| `Recall@k` | Fracao dos documentos relevantes recuperados no top-k |
| `MRR@k` | Mean Reciprocal Rank at k |
| `P@k` | Precision at k |

**Fluxo de calculo** (sobre todas as queries do dataset):
1. Qrels (ground truth) construidos como `ir_measures.Qrel(query_id, doc_id, relevance=1)` para cada doc relevante
2. Run (resultados recuperados) construidos como `ir_measures.ScoredDoc(query_id, doc_id, score)`
3. `ir_measures.calc_aggregate([nDCG@k, R@k, MRR@k, P@k], run, qrels)` calcula tudo de uma vez
4. Resultados agregados por media em `EvaluateUseCase` (`n = max(len(per_query), 1)`)

**Metricas de busca individuais** (por query no `SearchUseCase`): retornam `None` por padrao e sao preenchidas com valores reais pelo `_attach_ir_metrics()` no api_router quando ground truth existe para aquela query.

## Persistencia de estado dos encoders

Arquivos: `core/src/infrastructure/encoders/{classical,quantum,statistical}.py`, `core/src/infrastructure/api/deps.py`

Ao final do fit, cada encoder serializa seu estado (PCAs, SVD, min/max de angulos) em disco via `joblib`:

```
core/data/encoder_state/
├─ classical.joblib   # PCA(64) fitted
├─ quantum.joblib     # PCA_base(64) + PCA_angles(6) + PCA_final(64) + angle_min/max
└─ statistical.joblib # PCA(64) + TruncatedSVD(64) fitted
```

Na inicializacao do container, `_get_encoders()` em `deps.py` tenta carregar os arquivos automaticamente. Se presentes, os encoders ficam fitted sem precisar reindexar. O diretorio e configuravel via `ENCODER_STATE_DIR` (default: `/app/data/encoder_state`, que mapeia para `core/data/encoder_state/` no host pelo bind mount `./core:/app`).

## Medicao de tempo

Arquivo: `core/src/application/ir_use_cases.py` (`SearchUseCase._search_single`)

Metodologia identica para os tres pipelines via `time.perf_counter()`:

- `encode_time_ms` — intervalo entre inicio e fim do `encoder.encode(query)`
- `search_time_ms` — intervalo entre fim do encode e fim da busca no pgvector
- `total_time_ms` — soma de encode + search

Retornados em `metrics.encode_time_ms`, `metrics.search_time_ms`, `metrics.total_time_ms` na resposta da API.

Logs emitidos (uma linha por metrica):
```
[TIME] pipeline=classical encode_time_ms=5.2
[TIME] pipeline=classical search_time_ms=1.1
[TIME] pipeline=classical total_time_ms=6.3
```

## Logs de auditoria

Arquivo: `core/src/audit.py`

Dois mecanismos de log coexistem:

1. `audit_print(event, **payload)` — JSON estruturado com timestamp, usado para rastreamento completo:
   ```
   [AUDIT] {"ts": "...", "event": "search.pipeline.completed", ...}
   ```

2. `category_log(category, **payload)` — formato textual por categoria, obrigatorio pela especificacao experimental:
   ```
   [BASE] embedding_dim=384
   [PCA] input_dim=384 output_dim=64 pipeline=classical
   [PIPELINE classical] final_vector_dim=64
   [PIPELINE quantum] base_vector_dim=64 quantum_vector_dim=64 concat_dim=128 final_vector_dim=64
   [PIPELINE statistical] base_vector_dim=128 svd_input_dim=128 svd_output_dim=64
   [NORMALIZE] vector_norm=1.0
   [VECTOR SAMPLE classical] values=[0.1234, -0.0312, ...]
   [INDEX] dataset=beir/trec-covid doc_count=171332
   [SEARCH] pipeline=classical top_k=25 results=25
   [TIME] pipeline=classical encode_time_ms=5.2
   [TIME] pipeline=classical search_time_ms=1.1
   [TIME] pipeline=classical total_time_ms=6.3
   [METRICS INPUT] pipeline=classical run_docs=25 qrels_docs=3
   [METRICS RESULT] pipeline=classical nDCG@25=0.42 Recall@25=0.38 MRR=0.51 P@25=0.21
   ```

## Lote de indexacao

Arquivo: `core/src/application/ir_use_cases.py`

- Persistencia em lotes de 64 documentos por flush/upsert
- Reduz numero de commits e atualiza progresso do job de indexacao por lote
- Validacao explícita de dimensao antes do upsert: `len(vector) != VECTOR_DIM` lanca `ValueError`
- Amostra de vetor emitida no primeiro documento e a cada 100 documentos via `[VECTOR SAMPLE]`

## Teste de equivalencia (TOST)

Arquivo: `core/data/results/csv/equivalence_analysis.py`

### Pergunta

O teste de Wilcoxon bilateral (`statistical_analysis.py`, `paired_tests.csv`) nao
rejeitou a igualdade entre os pipelines `statistical` e `classical` no nDCG. Nao
rejeitar a hipotese nula e **ausencia de evidencia de diferenca**, e nao evidencia
de equivalencia: um teste de superioridade sem poder suficiente produz o mesmo
resultado que dois pipelines de fato equivalentes. Para afirmar equivalencia e
preciso um teste que a tenha como hipotese alternativa. Usa-se aqui o TOST
(*two one-sided tests*), aplicado ao par `statistical - classical` nas 4 metricas
(nDCG, Recall, MRR, Precision) e nos 4 cortes (k = 10, 25, 50, 100), totalizando
16 comparacoes sobre as mesmas 50 consultas pareadas por `query_id`.

### Margem de equivalencia

A margem e relativa, fixada em 5% da media do pipeline `classical` para aquela
metrica e corte:

```
delta = 0.05 x media(classical, metrica, k)
```

Justificativa: a regra de Sparck Jones (1974), reportada por Sanderson (2010,
p. 313), segundo a qual diferencas abaixo de 5% nao sao perceptiveis em avaliacao
de recuperacao de informacao. A margem e definida por essa regra **antes** de
olhar os resultados e nao e ajustada aos dados.

### Teste primario: TOST pareado com distribuicao t

Para cada (metrica, corte), com `d_i = statistical_i - classical_i`, `n = 50`,
`d_barra = media(d)`, `se = desvio(d, ddof=1) / raiz(n)` e `df = n - 1`:

```
p_lower = P(T > (d_barra + delta) / se)
p_upper = P(T < (d_barra - delta) / se)
p_tost  = max(p_lower, p_upper)
```

Conclui-se equivalencia quando `p_tost` corrigido por Holm fica abaixo de
`alfa = 0.05`. Registra-se tambem o intervalo de confianca t de 90%
(`d_barra +/- t_{0.95, df} x se`), cuja relacao com o TOST e direta: o teste
aceita equivalencia exatamente quando esse intervalo esta contido em
`(-delta, +delta)`.

**Por que t e nao Wilcoxon.** A margem de 5% esta definida sobre a **media** da
metrica. O teste de Wilcoxon testa a pseudo-mediana de Hodges-Lehmann, que e um
parametro diferente. Em metricas com muitas diferencas exatamente iguais a zero —
o caso do MRR, onde a maioria das consultas produz o mesmo primeiro acerto nos
dois pipelines — o Wilcoxon concentra-se no nucleo de empates e aceita
equivalencia mesmo quando a media esta fora da margem. Isso tornaria a conclusao
inconsistente com a margem declarada. Por isso o teste t e o primario; o TOST
com Wilcoxon e reportado apenas como registro suplementar e nao decide nada.

### Confirmacao por bootstrap de 90%

Para cada (metrica, corte) reamostram-se as 50 diferencas pareadas
(`BOOTSTRAP_RESAMPLES = 10000`, `BOOTSTRAP_SEED = 42`, gerador novo por
comparacao) e calcula-se o intervalo percentil de 90% da media. A coluna
`ci90_inside` indica se esse intervalo esta inteiramente dentro de
`(-delta, +delta)`, e `agree` indica se essa confirmacao concorda com a decisao
do teste t. Sao dois criterios com pressupostos distintos (o t assume
normalidade aproximada da media; o bootstrap nao) sobre o mesmo dado.

### Familia de Holm-Bonferroni

A correcao de Holm e aplicada **dentro de cada metrica, sobre os 4 cortes**
(familias de tamanho 4). Cortes diferentes da mesma metrica sao leituras
altamente correlacionadas do mesmo experimento, e a decisao de interesse e por
metrica; metricas distintas respondem a perguntas distintas e nao sao agrupadas
na mesma familia. A mesma correcao e aplicada, separadamente, ao TOST com
Wilcoxon e a cada configuracao da analise de sensibilidade.

### Sensibilidade a margem

Como qualquer conclusao de equivalencia depende da margem escolhida, o mesmo TOST
t e recalculado para margens absolutas (0.01, 0.02, 0.03, 0.05) e relativas
(2.5%, 5%, 10% da media do `classical`). Reporta-se ainda a **margem minima de
equivalencia** — a menor margem que o dado sustenta — como
`max(|limite inferior|, |limite superior|)` do IC de 90%, nas versoes t e
bootstrap, em valor absoluto e relativo a media do `classical`.

### Execucao

```
cd core/data/results/csv
python equivalence_analysis.py
```

O script reutiliza o carregador e as constantes de `statistical_analysis.py`
(`load_all`, `METRICS`, `K_VALUES`, `BOOTSTRAP_RESAMPLES`, `BOOTSTRAP_SEED`,
`ALPHA`, `holm_bonferroni`) e nao modifica nenhum arquivo pre-existente.
Dependencias: numpy e scipy.

### Arquivos gerados

Todos em `core/data/results/csv/`:

- `equivalence_tests.csv` — 16 linhas (4 metricas x 4 cortes) com medias, margem,
  p-valores do TOST t (bruto e Holm), decisao de equivalencia, IC t e bootstrap de
  90%, concordancia entre os dois criterios, TOST com Wilcoxon e margens minimas.
- `equivalence_sensitivity.csv` — 112 linhas (16 comparacoes x 7 margens
  alternativas) com `delta_type`, `delta_param`, `delta`, `p_tost`, `p_tost_holm`
  e `equivalent`.
- `equivalence_output.txt` — saida de console completa da execucao, incluindo as
  versoes de Python, numpy e scipy e os parametros `BOOTSTRAP_RESAMPLES`,
  `BOOTSTRAP_SEED` e `ALPHA`.
