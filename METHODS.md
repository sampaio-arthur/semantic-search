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
