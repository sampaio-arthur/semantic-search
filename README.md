# TCC - Quantum Comparative Retrieval

Aplicacao full-stack para comparar busca semantica em tres pipelines de vetorizacao sobre datasets BEIR:

| Pipeline | Transformacao | Coluna vetorial | Dim |
|---|---|---|---|
| **Classical** | SBERT → PCA(64) → L2 | `embedding_vector` | 64 |
| **Quantum** | SBERT → PCA_base(64) → PCA_angles(6) → QCircuit(64) → Hellinger → concat(128) → PCA_final(64) → L2 | `quantum_vector` | 64 |
| **Statistical** | SBERT → PCA(128) → TruncatedSVD(64) → L2 | `statistical_vector` | 64 |

Todos os tres pipelines compartilham o mesmo modelo base (`all-MiniLM-L6-v2`, 384 dim), o mesmo ranking por similaridade cosseno (pgvector) e as mesmas metricas (ir_measures). A unica variavel experimental e a **transformacao vetorial aplicada ao embedding SBERT**.

## Objetivo do projeto

Implementar e comparar tres fluxos de busca semantica sobre o mesmo dataset e com o mesmo criterio de ranking (similaridade cosseno), variando apenas a representacao vetorial:

- **Pipeline classico**: SBERT → reducao PCA(64) → L2
- **Pipeline quantico-inspirado**: SBERT → PCA residual → circuito PennyLane (StronglyEntanglingLayers) → Hellinger → concat → PCA_final → L2
- **Pipeline estatistico**: SBERT → PCA(128) → TruncatedSVD(64) → L2

Objetivo da comparacao:

- analisar diferencas de retrieval entre as tres representacoes vetoriais
- comparar proxies de custo (latencia)
- permitir avaliacao batch com ground truth (nDCG@k, Recall@k, MRR@k, P@k via `ir_measures`)

Inclui:

- backend FastAPI (Clean Architecture)
- frontend React + Vite
- PostgreSQL + pgvector
- autenticacao JWT
- chats persistidos
- cadastro de ground truth e avaliacao batch com metricas IR padrao

## Como o `.env.example` deve ser usado

O arquivo `.env.example` e a base das variaveis da aplicacao.

Use assim:

- copie `.env.example` para `.env`
- preencha os valores
- use esse `.env` no `docker compose` e/ou no run local

## Variaveis minimas recomendadas (`.env`)

Exemplo funcional para desenvolvimento local/Docker:

```env
APP_ENV=dev
APP_NAME=quantum-semantic-search

DB_SCHEME=postgresql+psycopg
DB_HOST=db
DB_PORT=5432
DB_NAME=tcc
DB_USER=tcc
DB_PASSWORD=tcc

JWT_SECRET=troque-esta-chave
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_MINUTES=10080

CLASSICAL_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIM=64
QUANTUM_N_QUBITS=6
PCA_INTERMEDIATE_DIM=128
SEED=42

PASSWORD_RESET_EXPIRE_MINUTES=30

REQUIRE_AUTH_FOR_INDEXING=true
REQUIRE_ADMIN_FOR_INDEXING=false

VITE_API_BASE_URL=http://localhost:8000
```

Invariante obrigatoria: `2 ** QUANTUM_N_QUBITS == VECTOR_DIM` (default: `2^6 = 64`).

## Pre-requisitos

### Opcao A (recomendada): Docker

- Docker
- Docker Compose

## Executando com Docker

1. Criar e configurar `.env` a partir de `.env.example`
2. Subir os servicos

```bash
docker compose up --build
```

Ou usando o Makefile:

```bash
make setup   # build + up
make up      # so subir
make logs    # ver logs
make down    # parar
make test    # rodar testes no container
```

Servicos:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- PostgreSQL/pgvector: `localhost:5432`

### Fluxo inicial para validar que tudo funciona

1. Acesse `http://localhost:5173`
2. Crie uma conta
3. Entre
4. Abra o chat e envie uma consulta
5. O frontend vai:
   - criar conversa
   - garantir indexacao do dataset `beir/trec-covid`
   - executar busca comparativa nos tres pipelines
   - exibir paineis com encode_time_ms, search_time_ms, total_time_ms, scores e metricas

Observacoes:

- A primeira indexacao pode demorar (modelo SBERT + dois passos de indexacao do corpus BEIR local)
- O job de indexacao assincrono usa polling em `/search/dataset/index/status`
- A indexacao e um processo de dois passos: (1) coleta todos os textos e faz o fit dos encoders; (2) transforma e persiste os vetores

## Dataset (BEIR)

Este projeto utiliza datasets do benchmark BEIR para avaliacao de busca semantica.

Os datasets NAO sao versionados no repositorio devido ao tamanho.

Exemplo usado:

- TREC-COVID (BEIR)

Fonte oficial:

- `https://github.com/beir-cellar/beir`

Estrutura esperada no projeto:

```text
core/data/beir/trec-covid/
├─ corpus.jsonl
├─ queries.jsonl
└─ qrels/test.tsv
```

O backend le os arquivos localmente (offline), sem download automatico.

Apos colocar os arquivos no local correto, o backend pode ser iniciado normalmente.

## Testes

```bash
# Rodar todos os testes dentro do container
make test

# Ou diretamente
docker compose exec core pytest

# Um arquivo especifico
docker compose exec core pytest tests/test_api_flow.py -v
```

## Documentacao adicional

- `DOCUMENTACAO.md` (visao tecnica aderente ao codigo)
- `ARCHITECTURE.md`
- `METHODS.md`
- `DB_SCHEMA.md`
- `DEPENDENCIES.md`

## Duvidas

- email: `arthurbarrasampaio@gmail.com`
