# Documentacao Tecnica - Semantic Search (estado atual do codigo)

## Objetivo

Comparar retrieval semantico em tres pipelines no mesmo dataset:

- **Classico** (`ClassicalPipelineEncoder` + cosseno em `embedding_vector`)
- **Quantico-inspirado** (`QuantumPipelineEncoder` + cosseno em `quantum_vector`)
- **Estatistico** (`StatisticalPipelineEncoder` + cosseno em `statistical_vector`)

O sistema tambem inclui:

- autenticacao JWT com refresh token
- chats persistidos (sem LLM; mensagens do assistant sao resumos/payloads de retrieval)
- cadastro de ground truth
- avaliacao batch com metricas IR padrao via `ir_measures`

## Arquitetura (aderente ao projeto)

### Camadas

- `core/src/domain`: entidades, enums, excecoes e ports
- `core/src/application`: casos de uso (auth, chats, IR)
- `core/src/infrastructure`: API FastAPI, banco, repositorios, encoders, metricas, seguranca, dataset provider
- `frontend`: SPA React/Vite consumindo rotas compativeis sem prefixo `/api`

### Servicos em runtime

- `core`: FastAPI + Uvicorn (porta 8000)
- `frontend`: Vite (porta 5173)
- `db`: PostgreSQL com imagem `pgvector/pgvector:pg16` (porta 5432)

### Persistencia vetorial

- PostgreSQL + pgvector via `VectorType` e operador `cosine_distance`
- Tres colunas vetoriais em `documents`: `embedding_vector vector(64)`, `quantum_vector vector(64)`, `statistical_vector vector(64)`

## Fluxo ponta a ponta

### 1) Dataset (BEIR local/offline)

- Provider implementado: `core/src/infrastructure/datasets/beir_local_provider.py`
- Fonte: arquivos locais no formato BEIR (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`)
- Sem download automatico
- Estrutura esperada: `core/data/beir/<dataset_name>/...`
- Dataset padrao: `beir/trec-covid`

### 2) Indexacao

Caso de uso: `IndexDatasetUseCase`

Passos reais:

1. Busca metadados do dataset no provider
2. **Passo 1**: Itera documentos do corpus, coleta textos
3. **Fit**: Encode em lote via `SharedSbertBase` (SBERT) → ajusta `ClassicalPipelineEncoder`, `QuantumPipelineEncoder` (3 PCAs + passo do circuito) e `StatisticalPipelineEncoder` (PCA + SVD)
4. **Passo 2**: Transforma cada documento com os tres encoders → upsert em `documents` em lotes de 64
5. Persiste todas as `queries` e `qrels` (split `test`) no banco sem filtro de exclusao
6. Persiste snapshot em `dataset_snapshots` (doc_ids, queries, metadados)

Observacoes:

- `POST /api/index` executa indexacao sincrona
- O frontend usa a rota compativel `POST /search/dataset/index`, que dispara job em thread + polling em `/search/dataset/index/status`
- O job assincrono pode pular reindexacao (`already_indexed`) apenas se snapshot, contagem de docs **e** encoders fitted coincidirem — sem encoders fitted, reindexacao e obrigatoria
- Apos o fit, o estado dos encoders e salvo em disco (`core/data/encoder_state/`) para sobreviver a reinicializacoes do container

### 3) Busca

Caso de uso: `SearchUseCase`

Modos suportados:

- `classical`
- `quantum`
- `statistical`
- `compare`

**Pipeline classico**:

1. Encode da query com `ClassicalPipelineEncoder`
2. Busca em `documents.embedding_vector`
3. Score por similaridade cosseno, ordenacao descrescente

**Pipeline quantico-inspirado**:

1. Encode da query com `QuantumPipelineEncoder`
2. Busca em `documents.quantum_vector`
3. Score por similaridade cosseno, ordenacao descrescente

**Pipeline estatistico**:

1. Encode da query com `StatisticalPipelineEncoder`
2. Busca em `documents.statistical_vector`
3. Score por similaridade cosseno, ordenacao descrescente

**Medicao de tempo (identica para os tres pipelines)**:

- `encode_time_ms` — tempo de geracao do vetor da query (`time.perf_counter()`)
- `search_time_ms` — tempo da busca no pgvector
- `total_time_ms` — soma de encode + search

**Modo `compare`** retorna tambem:

- `comparison.classical`, `comparison.quantum`, `comparison.statistical`
- `comparison_metrics` com sobreposicao entre os tres top-k (intersecoes por pares e total)

### 4) Chat persistido

Fluxo usado pelo frontend (`frontend/src/pages/Chat.tsx`):

1. Cria conversa (se ainda nao houver uma ativa)
2. Salva mensagem do usuario
3. Garante indexacao do dataset `beir/trec-covid` (com polling)
4. Executa busca comparativa nos tres pipelines
5. Salva mensagem `assistant` textual resumindo resultados
6. Armazena ultima resposta completa em `localStorage` para renderizar os paineis

Tambem existe persistencia estruturada pelo backend:

- `POST /api/search` com `chat_id` salva um payload JSON de retrieval como mensagem `assistant`

## Avaliacao e gabaritos (estado real)

### Ground truth

Tabelas usadas:

- `queries` (`dataset`, `split`, `query_id`, `query_text`, `ideal_answer`)
- `qrels` (`dataset`, `split`, `query_id`, `doc_id`, `relevance`)

Para compatibilidade com o fluxo atual de avaliacao, o repositorio monta `relevant_doc_ids` a partir de `qrels` com `relevance > 0`. `list_by_dataset()` retorna todas as queries do dataset sem filtro de exclusao.

### Cadastro de gabaritos

- API canonica: `POST /api/ground-truth`
- API compativel usada pelo frontend: `POST /benchmarks/labels`

Importante sobre `ideal_answer`:

- O frontend envia `ideal_answer`
- A rota compativel (`POST /benchmarks/labels`) persiste `ideal_answer` quando o campo e enviado (coluna `ideal_answer TEXT NULL` na tabela `queries`)
- Se `relevant_doc_ids` nao forem enviados, o backend infere um ground truth inicial usando busca classica top-5 e salva esses `doc_ids`

### Avaliacao batch

Caso de uso: `EvaluateUseCase`

Fluxo:

1. Le gabaritos do dataset via `list_by_dataset()` (retorna todas as queries do dataset)
2. Executa busca por `query_text` no(s) pipeline(s) com `top_k` configuravel (10, 25, 50 ou 100; padrao 25)
3. Calcula metricas IR por query via `IrMeasuresAdapter`
4. Se `ideal_answer` estiver definido na query, calcula `answer_similarity` via `AnswerSimilarityService`
5. Captura timing por query (`encode_time_ms`, `search_time_ms`, `total_time_ms`)
6. Agrega medias por pipeline (metricas IR + `mean_answer_similarity` + `mean_encode_time_ms`, `mean_search_time_ms`, `mean_total_time_ms`)

A avaliacao batch pode ser executada de forma assincrona via `EvaluationJobRegistry` (`infrastructure/api/evaluation_jobs.py`), com progresso rastreado por query e pipeline. O frontend (`BatchEvaluation.tsx`) faz polling no status, exibe graficos comparativos (Recharts), permite selecionar o valor de k (10, 25, 50 ou 100) e exportar os resultados em CSV.

Metricas de retrieval calculadas (via biblioteca `ir_measures`, sem implementacao manual):

- `nDCG@k` — Normalized Discounted Cumulative Gain
- `Recall@k`
- `MRR@k` — Mean Reciprocal Rank
- `P@k` — Precision at k

Metrica de avaliacao semantica:

- `answer_similarity` — similaridade cosseno entre o embedding SBERT dos top-3 documentos recuperados (concatenados) e o embedding do `ideal_answer`. Calculada por `AnswerSimilarityService` (`infrastructure/metrics/answer_similarity.py`) usando o mesmo modelo `all-MiniLM-L6-v2` compartilhado pelos encoders.

**Nota sobre top_k e metricas**: o valor de `top_k` e configuravel pelo usuario entre 10, 25, 50 e 100 (`ALLOWED_TOP_K` em `domain/ir.py`, padrao `DEFAULT_TOP_K = 25`). Controla tanto o numero de documentos retornados pela query SQL quanto o conjunto sobre o qual as metricas IR sao calculadas. O frontend permite selecionar o valor tanto na busca individual quanto na avaliacao batch.

## Avaliacao semantica (ideal_answer / answer_similarity)

O sistema implementa avaliacao semantica da qualidade das respostas recuperadas com base em uma resposta ideal cadastrada por query.

### Persistencia

- A coluna `ideal_answer TEXT NULL` existe na tabela `queries` (adicionada via migration `001_add_ideal_answer_to_queries.py`)
- O campo e persistido pelo `SqlAlchemyGroundTruthRepository.upsert()` quando nao-nulo
- O dominio representa o campo em `GroundTruth.ideal_answer: str | None`

### Edicao via frontend

- Pagina `EvaluationQueries.tsx`: lista todas as queries do dataset com botao para editar o `ideal_answer` inline; exibe contador "N/total com gabarito"
- Pagina `Benchmarks.tsx`: formulario manual para criar benchmark labels com `ideal_answer`
- Ambas as paginas chamam `api.upsertBenchmarkLabel()` que envia `ideal_answer` via `POST /benchmarks/labels`

### Calculo de answer_similarity

Servico: `AnswerSimilarityService` (`infrastructure/metrics/answer_similarity.py`)

- Usa o mesmo modelo SBERT (`all-MiniLM-L6-v2`) compartilhado pelos encoders via `SharedSbertBase`
- Calcula similaridade cosseno entre dois textos: `cosine_similarity(embed(text_a), embed(text_b))`
- Trata caso de vetor nulo (retorna `0.0` com log explicito)
- Retorna valor em `[-1, 1]` arredondado a 4 casas decimais

### Construcao do texto de resposta recuperada

Para calcular `answer_similarity`, o sistema:

1. Concatena os textos dos top-3 documentos recuperados: `" ".join(doc.text for doc in results[:3])`
2. Computa `cosine_similarity(embed(top3_text), embed(ideal_answer))`

O metodo e identico para todos os pipelines.

### Integracao no fluxo de busca

- `_attach_answer_similarity()` em `api_router.py`: chamado apos cada busca; verifica se existe ground truth com `ideal_answer` para a query e calcula a similaridade por pipeline
- `EvaluateUseCase`: calcula `answer_similarity` por query em avaliacao batch; agrega como `mean_answer_similarity`
- `ComparisonPanel.tsx`: exibe linha "Answer Similarity" na tabela de comparacao dos tres pipelines

### Logs

```
[SEMANTIC EVAL] query_id=... pipeline=classical similarity=0.7341
```

Tambem emitido via `audit_print()`:

```
{"event": "answer_similarity.compute.completed", "similarity": 0.7341}
```

---

## API (resumo de operacao)

### Rotas canonicas (`/api/*`)

- `GET /api/health`
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/forgot-password`
- `POST /api/auth/reset-password`
- `POST /api/auth/refresh`
- `POST|GET /api/chats`
- `GET|PATCH|DELETE /api/chats/{chat_id}`
- `POST /api/chats/{chat_id}/messages`
- `POST /api/index`
- `POST /api/search`
- `POST /api/ground-truth`
- `POST /api/evaluate`

### Rotas compativeis (sem `/api`) expostas para o frontend atual

- `/auth/*`
- `/conversations*`
- `/search/dataset/index`
- `/search/dataset/index/status`
- `/search/dataset`
- `/datasets*`
- `/benchmarks/labels*`
- `/benchmarks/evaluate/start` — inicia avaliacao batch assincrona
- `/benchmarks/evaluate/status` — polling do status da avaliacao batch
- `/evaluation/queries` — lista queries BEIR com ideal answers

Rota descontinuada mantida apenas para retorno de erro:

- `POST /search/file` → `410 Gone`

## Seguranca

- Senhas com `passlib` (`bcrypt`)
- JWT access + refresh com `python-jose`
- `OAuth2PasswordBearer` no FastAPI
- Chats, indexacao e busca exigem usuario autenticado
- `REQUIRE_ADMIN_FOR_INDEXING` pode exigir admin na indexacao

## Configuracao (variaveis reais do backend)

Arquivo: `core/src/infrastructure/config.py`

### Aplicacao

- `APP_ENV` (default: `dev`)
- `APP_NAME` (default: `quantum-semantic-search`)
- `CORS_ORIGINS` (default: `["http://localhost:5173"]`)

### Banco

- `DB_SCHEME`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `DATABASE_URL` (override opcional)

### Auth/JWT

- `JWT_SECRET` (deve ser trocado em producao)
- `JWT_ALGORITHM` (default: `HS256`)
- `ACCESS_TOKEN_EXPIRE_MINUTES` (default: `30`)
- `REFRESH_TOKEN_EXPIRE_MINUTES` (default: `10080` = 7 dias)
- `PASSWORD_RESET_EXPIRE_MINUTES` (default: `30`)

### Retrieval / encoders

- `CLASSICAL_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `VECTOR_DIM` (default: `64`) — dimensao final dos tres pipelines
- `QUANTUM_N_QUBITS` (default: `6`) — invariante: `2^QUANTUM_N_QUBITS == VECTOR_DIM`
- `PCA_INTERMEDIATE_DIM` (default: `128`) — dim intermediaria do pipeline estatistico; deve ser maior que `VECTOR_DIM` para que o `TruncatedSVD` realize reducao real de dimensionalidade (se igual, o SVD degenera para rotacao ortogonal e o resultado seria identico ao pipeline classico sob similaridade cosseno)
- `SEED` (default: `42`) — semente para PCAs, SVD e pesos do circuito
- `ENCODER_STATE_DIR` (default: `/app/data/encoder_state`) — diretorio onde o estado fitted dos encoders e persistido em disco

### Controle de acesso

- `REQUIRE_AUTH_FOR_INDEXING` (default: `true`)
- `REQUIRE_ADMIN_FOR_INDEXING` (default: `false`)

### Frontend

- `VITE_API_BASE_URL` (consumido em `frontend/src/lib/api.ts`)

## Banco de dados (resumo)

Tabelas principais:

- `users`
- `password_resets`
- `chats`
- `chat_messages`
- `documents` (com tres colunas vetoriais: `embedding_vector vector(64)`, `quantum_vector vector(64)`, `statistical_vector vector(64)`)
- `queries`
- `qrels`
- `dataset_snapshots`

Detalhes em `DB_SCHEMA.md`.

## Limitacoes atuais

- Pipeline quantico e simulado (PennyLane `default.qubit`), nao hardware quantico real
- A primeira indexacao pode ser demorada em datasets BEIR grandes (SBERT + dois passos sobre o corpus inteiro); reindexacoes subsequentes apos restart sao evitadas pelo estado persistido em disco
- Dataset BEIR deve ser colocado manualmente em `core/data/beir/` (sem download automatico)

## Referencias de implementacao

- Busca e avaliacao: `core/src/application/ir_use_cases.py`
- Metricas: `core/src/infrastructure/metrics/ir_measures_adapter.py`
- Encoders: `core/src/infrastructure/encoders/base.py`, `classical.py`, `quantum.py`, `statistical.py`
- Rotas: `core/src/infrastructure/api/routers/api_router.py`
- Job de avaliacao assincrono: `core/src/infrastructure/api/evaluation_jobs.py`
- Dashboard de avaliacao batch: `frontend/src/pages/BatchEvaluation.tsx`
- Frontend API client: `frontend/src/lib/api.ts`
