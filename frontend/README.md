# Quantum Search Frontend

Interface web do projeto de busca semantica comparativa.

O frontend:

- autentica usuarios
- cria e persiste conversas
- dispara indexacao do dataset BEIR local (`beir/trec-covid`)
- executa busca comparativa entre os tres pipelines:
  - pipeline classico: SBERT → PCA(64) → L2
  - pipeline quantico-inspirado: SBERT → PCA_base → QCircuit → Hellinger → PCA_final → L2 (PennyLane)
  - pipeline estatistico: SBERT → PCA(128) → TruncatedSVD(64) → L2
- exibe resultados, scores e latencia dos tres pipelines
- permite cadastrar `ideal_answer` por query (paginas Benchmarks e EvaluationQueries)
- exibe metrica de avaliacao semantica `answer_similarity` na comparacao

## Executar localmente

```bash
npm install
npm run dev
```

Aplicacao padrao em `http://localhost:5173` quando executada via Docker Compose deste projeto.

## Build

```bash
npm run build
```

## Testes

```bash
npm run test
```
