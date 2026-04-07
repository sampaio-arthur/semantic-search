# DB Schema

## Banco de dados

PostgreSQL com extensao `pgvector` (`pgvector/pgvector:pg16`).

Todos os tres pipelines armazenam vetores de dimensao 64 (`VECTOR_DIM=64`) em colunas separadas na tabela `documents`.

## Tabelas

### `users`

- `id`
- `email` (UNIQUE)
- `password_hash`
- `name`
- `is_active`
- `is_admin`
- `created_at`

### `password_resets`

- `id`
- `user_id` (FK → `users`)
- `token_hash`
- `expires_at`
- `used_at`
- `created_at`

### `chats`

- `id`
- `user_id` (FK → `users`)
- `title`
- `created_at`
- `updated_at`
- `deleted_at` (soft delete)

### `chat_messages`

- `id`
- `chat_id` (FK → `chats`)
- `role` (`user|assistant|system`)
- `content` (texto; payload JSON serializado e permitido para resultado de retrieval)
- `created_at`

### `documents`

- `id`
- `dataset`
- `doc_id`
- `title` (opcional)
- `text`
- `metadata` (JSON/JSONB)
- `embedding_vector vector(64)` — Pipeline classico (SBERT → PCA(64) → L2)
- `quantum_vector vector(64)` — Pipeline quantico-inspirado (SBERT → PCA_base → circuito → Hellinger → PCA_final → L2)
- `statistical_vector vector(64)` — Pipeline estatistico (SBERT → PCA(128) → TruncatedSVD(64) → L2)

Constraint:

- `UNIQUE(dataset, doc_id)`

### `queries`

- `id`
- `dataset`
- `split`
- `query_id`
- `query_text`
- `user_id` (opcional)
- `created_at`

Constraint:

- `UNIQUE(dataset, split, query_id)`

### `qrels`

- `id`
- `dataset`
- `split`
- `query_id`
- `doc_id`
- `relevance` (int)

Constraint:

- `UNIQUE(dataset, split, query_id, doc_id)`

### `dataset_snapshots`

- `id`
- `dataset_id` (UNIQUE)
- `name`
- `provider`
- `description`
- `source_url` (opcional)
- `reference_urls` (JSON/JSONB)
- `max_docs` / `max_queries` (recorte persistido)
- `document_count` / `query_count`
- `document_ids` (JSON/JSONB; lista exata de docs indexados)
- `queries` (JSON/JSONB; snapshot das queries e `relevant_doc_ids`)
- `created_at`
- `updated_at`

Constraint:

- `UNIQUE(dataset_id)`

## Vetores

- PostgreSQL + pgvector via `VectorType` (`pgvector.sqlalchemy.Vector`)
- Todos os vetores tem dimensao 64 (configuravel via `VECTOR_DIM`, default `64`)
- Busca por similaridade cosseno: `cosine_distance` do pgvector
- `score = 1 - cosine_distance(query_vector, doc_vector)`
- Todos os vetores sao L2-normalizados antes do armazenamento
