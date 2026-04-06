# Architecture

## Estilo

Clean Architecture com separacao em:

- `domain/`: entidades, value objects, regras e ports (Protocols)
- `application/`: casos de uso (orquestracao)
- `infrastructure/`: adapters concretos (DB, JWT, bcrypt, encoders, metricas, dataset)
- `infrastructure/api/`: delivery via FastAPI

## Principais Casos de Uso

- **Auth**: `SignUpUseCase`, `SignInUseCase`, `RequestPasswordResetUseCase`, `ConfirmPasswordResetUseCase`, `RefreshTokenUseCase`
- **Chats**: `CreateChatUseCase`, `ListChatsUseCase`, `GetChatUseCase`, `AddMessageUseCase`, `RenameChatUseCase`, `DeleteChatUseCase`
- **IR**: `IndexDatasetUseCase`, `SearchUseCase`, `UpsertGroundTruthUseCase`, `EvaluateUseCase`, `BuildAssistantRetrievalMessage`

## Encoders

Todos os tres encoders vivem em `infrastructure/encoders/` e compartilham `SharedSbertBase` (`base.py`) — instancia unica do modelo SBERT cacheada globalmente.

**Padrão fit/transform**: todos os encoders exigem `fit(raw_embeddings)` antes de `encode()`/`transform()`. O fit ocorre uma vez durante a indexacao (dois passos: coleta todos os textos → encode em lote pelo SBERT → faz o fit dos tres encoders → transforma + upsert).

| Encoder | Arquivo | Transformacao |
|---|---|---|
| `ClassicalPipelineEncoder` | `encoders/classical.py` | SBERT(384) → PCA(64) → L2 |
| `QuantumPipelineEncoder` | `encoders/quantum.py` | SBERT(384) → PCA_base(64) → PCA_angles(6) → AngleEmbedding+StronglyEntanglingLayers → probs(64) → sqrt(probs) → concat(128) → PCA_final(64) → L2 |
| `StatisticalPipelineEncoder` | `encoders/statistical.py` | SBERT(384) → PCA(128) → TruncatedSVD(64) → L2 |

Os singletons de encoder (`_classical_encoder`, `_quantum_encoder`, `_statistical_encoder`) sao criados uma vez na primeira requisicao e reutilizados. O estado fitted (PCAs, SVD) persiste entre requisicoes e tambem sobrevive a reinicializacoes do container: apos o fit, cada encoder serializa seu estado em disco via `joblib` (`save_state`/`load_state`), e na inicializacao do container o estado e recarregado automaticamente. Diretorio configuravel via `ENCODER_STATE_DIR` (default `core/data/encoder_state/` no host).

**Nota sobre o pipeline statistical**: usa `PCA(n=128)` seguido de `TruncatedSVD(n=64)`. A variavel `PCA_INTERMEDIATE_DIM` controla a dimensao intermediaria (default `128`). E obrigatorio que `PCA_INTERMEDIATE_DIM > VECTOR_DIM`: se iguais, o SVD degenera para uma rotacao ortogonal e o resultado sob similaridade cosseno seria identico ao pipeline classico.

## Principios importantes aplicados

- Mesma funcao de encoding para indexacao e busca dentro de cada pipeline (sem deriva entre os espacos)
- Nao mistura espacos vetoriais (colunas separadas no banco: `embedding_vector`, `quantum_vector`, `statistical_vector`)
- Todos os tres pipelines produzem vetores de dimensao identica (dim=64)
- Comparacao metodologica entre representacoes usando o mesmo criterio de ranking (cosseno via pgvector)
- Metricas IR calculadas pela biblioteca `ir_measures` (padrao da area)
- Dependencias de framework ficam fora do dominio
- Rotas canonicas (`/api/*`) e rotas compativeis (sem prefixo) coexistem sem quebrar o front

## Injecao de Dependencias

`deps.py:build_services()` conecta todos os adapters concretos ao dataclass `Services`. Cada requisicao recebe um `Services` via `get_services()` (FastAPI `Depends`).

## Dois Roteadores FastAPI

- `router` (prefixo `/api`) — rotas canonicas (ex: `/api/search`, `/api/chats`)
- `compat_router` (sem prefixo) — rotas legadas para compatibilidade com o frontend atual (ex: `/search/dataset`, `/conversations`)

## Fluxo de Indexacao (dois passos)

1. **Passo 1**: itera todos os documentos do corpus, coleta textos
2. **Fit**: encode em lote via SBERT → faz o fit dos tres encoders (ClassicalPipelineEncoder, QuantumPipelineEncoder, StatisticalPipelineEncoder)
3. **Passo 2**: transforma cada documento com os tres encoders → upsert no banco (lotes de 64 docs)
4. Persiste todas as queries e qrels (ground truth do split `test`) sem filtro de exclusao
5. Upsert do snapshot de metadados do dataset

## Fluxo de Busca

1. API recebe `query` + `dataset` + `mode` (`classical` | `quantum` | `statistical` | `compare`). O `top_k` e configuravel (10, 25, 50 ou 100, padrao 25 via `DEFAULT_TOP_K` em `domain/ir.py`)
2. `SearchUseCase` chama `encode()` do(s) encoder(s) correspondente(s)
3. Repositorio consulta apenas a coluna vetorial correspondente (`embedding_vector`, `quantum_vector` ou `statistical_vector`)
4. Retorna ranking com score (cosine similarity = 1 - cosine_distance)
5. Opcional: enriquecimento com metricas IR via `IrMeasuresAdapter` se ground truth (`qrels`) existir para a query
6. Opcional: calculo de `answer_similarity` via `AnswerSimilarityService` se `ideal_answer` estiver definido na query — concatena top-3 docs recuperados e computa similaridade cosseno SBERT com o `ideal_answer`
7. Opcional: persistencia de mensagem `assistant` no chat

## Avaliacao Semantica (answer_similarity)

Servico: `AnswerSimilarityService` (`infrastructure/metrics/answer_similarity.py`)

- Calcula `cosine_similarity(embed(top3_text), embed(ideal_answer))` usando o mesmo modelo SBERT compartilhado
- `top3_text` = concatenacao dos textos dos tres primeiros documentos recuperados
- `ideal_answer` persistido na coluna `queries.ideal_answer TEXT NULL`; editavel por query via `EvaluationQueries.tsx`
- Retorna float em `[-1, 1]` arredondado a 4 casas decimais
- Integrado em busca individual (`_attach_answer_similarity()`) e avaliacao batch (`EvaluateUseCase` → `mean_answer_similarity`)
- Log emitido: `[SEMANTIC EVAL] query_id=... pipeline=... similarity=...`

## Exclusao de queries (desativada)

O mecanismo de exclusao de queries foi desativado. `EXCLUDED_QUERY_TEXTS` em `domain/excluded_queries.py` e um `frozenset` vazio, `is_excluded_query()` retorna sempre `False`, e `list_by_dataset()` nao aplica nenhum filtro de exclusao. O sistema usa todas as queries do dataset (ate 50 para trec-covid apos reindexacao).

## Observacoes

- A busca usa PostgreSQL + pgvector com operador `cosine_distance`
- `score = 1 - cosine_distance(query_vector, doc_vector)`
- Os pipelines diferem apenas na representacao vetorial, nao no mecanismo de ranking
- Indexacao assincrona via `index_job_registry` com polling pelo frontend
- Avaliacao batch assincrona via `evaluation_job_registry` (`infrastructure/api/evaluation_jobs.py`) com polling pelo frontend (`BatchEvaluation.tsx`), seletor de k e exportacao CSV
- `top_k` configuravel (10, 25, 50, 100) via `ALLOWED_TOP_K` em `domain/ir.py`, padrao 25 (`DEFAULT_TOP_K`)
