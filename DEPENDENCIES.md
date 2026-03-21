# Dependencies (estado atual e onde cada uma e usada)

Este arquivo descreve as dependencias declaradas no projeto e o uso real no codigo atual.

## Backend (`core/requirements.txt`)

### Runtime principal (usadas diretamente)

| Dependencia | Onde e usada | Observacoes |
| --- | --- | --- |
| `fastapi` | `core/src/infrastructure/api/fastapi_app.py`, `core/src/infrastructure/api/routers/api_router.py`, `core/src/infrastructure/api/deps.py` | API HTTP, DI (`Depends`), schemas de request/response, CORS, `OAuth2PasswordBearer` |
| `uvicorn[standard]` | `docker-compose.yml` (comando do servico `core`) | Servidor ASGI para subir a API |
| `sqlalchemy` | `core/src/infrastructure/db/models.py`, `core/src/infrastructure/db/session.py`, `core/src/infrastructure/repositories/sqlalchemy_repositories.py` | ORM, sessoes, queries, modelos e persistencia |
| `psycopg[binary]` | usado indiretamente por `sqlalchemy` quando `DATABASE_URL` aponta para PostgreSQL | Driver PostgreSQL |
| `alembic` | `core/alembic/*`, `core/alembic.ini` | Estrutura de migracoes (o app sobe com `Base.metadata.create_all`) |
| `pydantic` | `core/src/infrastructure/api/schemas.py`, `core/src/infrastructure/config.py` | Schemas da API e base de configuracao |
| `pydantic-settings` | `core/src/infrastructure/config.py` | Leitura de `.env` via `BaseSettings` |
| `python-jose[cryptography]` | `core/src/infrastructure/security/adapters.py` | Criacao/validacao de JWT access e refresh |
| `passlib[bcrypt]` | `core/src/infrastructure/security/adapters.py` | Hash e verificacao de senha via `CryptContext` |
| `bcrypt` | backend do `passlib` | Algoritmo de hash efetivo |
| `python-multipart` | `core/src/infrastructure/api/routers/api_router.py` | Necessario para `OAuth2PasswordRequestForm` |
| `numpy` | `core/src/infrastructure/encoders/quantum.py`, `core/src/infrastructure/encoders/statistical.py` | Vetores/arrays, operacoes matriciais nos encoders |
| `scipy` | `core/src/infrastructure/encoders/statistical.py` | `TruncatedSVD` via `sklearn` usa scipy internamente; presente como dependencia direta |
| `scikit-learn` | `core/src/infrastructure/encoders/classical.py`, `core/src/infrastructure/encoders/quantum.py`, `core/src/infrastructure/encoders/statistical.py` | `PCA` e `TruncatedSVD` usados nos tres pipelines; `joblib` (dependencia transitiva do sklearn) usado para serializar/deserializar estado fitted em disco |
| `pgvector` | `core/src/infrastructure/db/models.py` | Tipo vetorial no PostgreSQL (`pgvector.sqlalchemy.Vector`) |
| `sentence-transformers` | `core/src/infrastructure/encoders/base.py` (`SharedSbertBase`) | Modelo SBERT compartilhado pelos tres encoders (`all-MiniLM-L6-v2`) |
| `pennylane` | `core/src/infrastructure/encoders/quantum.py` | Encoder quantico-inspirado (`qml.qnode`, `AngleEmbedding`, `StronglyEntanglingLayers`, `qml.probs`) |
| `autoray` | dependencia de compatibilidade do ecossistema PennyLane (pinada em `requirements.txt`) | Mantida para evitar quebra de versao com `pennylane==0.39.0` |
| `ir-measures` | `core/src/infrastructure/metrics/ir_measures_adapter.py` | Metricas IR padrao: `nDCG@k`, `Recall@k`, `MRR@k`, `P@k` via `calc_aggregate()` |

### Testes e suporte de teste (declaradas no mesmo `requirements.txt`)

| Dependencia | Onde e usada | Observacoes |
| --- | --- | --- |
| `pytest` | `core/tests/test_api_flow.py`, `core/pytest.ini` | Runner de testes |
| `httpx` | usado pelo `fastapi.testclient`/Starlette | Dependencia de suporte para `TestClient` em testes |

## Frontend (`frontend/package.json`)

### Runtime usado diretamente no app atual (rotas/paginas/fluxo principal)

| Dependencia | Onde e usada | Observacoes |
| --- | --- | --- |
| `react` | toda a app (`frontend/src/*`) | Base da SPA |
| `react-dom` | `frontend/src/main.tsx` | Render da aplicacao |
| `react-router-dom` | `frontend/src/App.tsx`, `frontend/src/pages/*.tsx` | Rotas (`/auth`, `/chat`, `/benchmarks`) |
| `@tanstack/react-query` | `frontend/src/App.tsx` | `QueryClientProvider` (infra preparada; fluxo atual ainda usa chamadas manuais no `api.ts`) |
| `lucide-react` | paginas e componentes de chat/auth (`frontend/src/pages/*.tsx`, `frontend/src/components/chat/*.tsx`) | Icones UI |
| `sonner` | `frontend/src/components/ui/sonner.tsx`, `frontend/src/App.tsx` | Toaster de notificacoes |
| `next-themes` | `frontend/src/components/ui/sonner.tsx` | Le tema para o Sonner (`useTheme`) |
| `clsx` | `frontend/src/lib/utils.ts` | Composicao de classes CSS |
| `tailwind-merge` | `frontend/src/lib/utils.ts` | Merge de classes Tailwind |
| `class-variance-authority` | `frontend/src/components/ui/button.tsx`, `.../toast.tsx` e outros componentes UI | Variantes de componentes shadcn |
| `@radix-ui/react-slot` | `frontend/src/components/ui/button.tsx` (e outros) | Padrao `asChild` em componentes shadcn |
| `@radix-ui/react-avatar` | `frontend/src/components/ui/avatar.tsx` | Avatar usado no chat/sidebar |
| `@radix-ui/react-scroll-area` | `frontend/src/components/ui/scroll-area.tsx` | Scroll da sidebar |
| `@radix-ui/react-tooltip` | `frontend/src/components/ui/tooltip.tsx`, `frontend/src/App.tsx` | Provider global de tooltip |
| `@radix-ui/react-toast` | `frontend/src/components/ui/toast.tsx`, `frontend/src/components/ui/toaster.tsx` | Sistema de toast (alem do Sonner) |
| `@radix-ui/react-label` | `frontend/src/components/ui/label.tsx` | Labels do formulario de auth |

### Dependencias de runtime presentes por causa do kit UI (`frontend/src/components/ui/*.tsx`)

Estas dependencias aparecem em componentes shadcn gerados e estao no repositorio, mesmo que varias nao sejam usadas pelas paginas atuais:

- `@hookform/resolvers`
- `@radix-ui/react-accordion`
- `@radix-ui/react-alert-dialog`
- `@radix-ui/react-aspect-ratio`
- `@radix-ui/react-checkbox`
- `@radix-ui/react-collapsible`
- `@radix-ui/react-context-menu`
- `@radix-ui/react-dialog`
- `@radix-ui/react-dropdown-menu`
- `@radix-ui/react-hover-card`
- `@radix-ui/react-menubar`
- `@radix-ui/react-navigation-menu`
- `@radix-ui/react-popover`
- `@radix-ui/react-progress`
- `@radix-ui/react-radio-group`
- `@radix-ui/react-select`
- `@radix-ui/react-separator`
- `@radix-ui/react-slider`
- `@radix-ui/react-switch`
- `@radix-ui/react-tabs`
- `@radix-ui/react-toggle`
- `@radix-ui/react-toggle-group`
- `cmdk`
- `embla-carousel-react`
- `input-otp`
- `react-day-picker`
- `react-hook-form`
- `react-resizable-panels`
- `recharts` (**usado ativamente** em `BatchEvaluation.tsx` para graficos de metricas e latencia)
- `vaul`
- `zod`
- `date-fns`

O app atual usa apenas uma parte pequena desse conjunto nas paginas de Auth/Chat/Benchmarks.

### Dev/build/test (frontend)

| Dependencia | Onde e usada | Observacoes |
| --- | --- | --- |
| `vite` | `frontend/package.json` scripts, `frontend/vite.config.ts` | Bundler/dev server |
| `@vitejs/plugin-react-swc` | `frontend/vite.config.ts`, `frontend/vitest.config.ts` | Plugin React + SWC |
| `typescript` | `frontend/tsconfig*.json` | Tipagem TS |
| `eslint`, `@eslint/js`, `typescript-eslint`, `eslint-plugin-react-hooks`, `eslint-plugin-react-refresh`, `globals` | `frontend/eslint.config.js` | Lint |
| `vitest` | `frontend/vitest.config.ts`, `frontend/src/test/example.test.ts` | Testes frontend |
| `jsdom` | `frontend/vitest.config.ts` (`environment: jsdom`) | Ambiente DOM de teste |
| `@testing-library/react`, `@testing-library/jest-dom` | `frontend/src/test/setup.ts` | Testes de componentes |
| `tailwindcss`, `postcss`, `autoprefixer` | `frontend/tailwind.config.ts`, `frontend/postcss.config.js` | Pipeline CSS |
| `tailwindcss-animate` | `frontend/tailwind.config.ts` | Plugin de animacoes |
| `@tailwindcss/typography` | instalado como devDependency | Nao aparece referenciado no config atual |
| `@types/node`, `@types/react`, `@types/react-dom` | configs TS/compilacao | Tipos |

## Infra / Runtime externo (fora de manifests de linguagem)

| Componente | Onde e usado | Observacoes |
| --- | --- | --- |
| `Docker Compose` | `docker-compose.yml` | Orquestra `core`, `frontend`, `db` |
| `pgvector/pgvector:pg16` | servico `db` em `docker-compose.yml` | Postgres com extensao `vector` |
| `Makefile` | `Makefile` | Atalhos de operacao (`setup`, `up`, `down`, `logs`, `test`, `restart`) |
