# Projeto Final - Pipeline de Visao Computacional

Pipeline de embeddings (MobileNetV2) + PostgreSQL/pgvector + avaliacao de busca semantica (Issue 4).

**Dois modos:** desenvolvimento **offline no Mac** (`--mode local`) e producao **na EC2** (`--mode aws` / S3). Detalhes: [docs/offline_and_aws.md](docs/offline_and_aws.md).

## Estrutura principal

- `pipeline_config.py`: modo local vs AWS e caminhos padrao.
- `preprocess.py`: download Kaggle + imagens em `data/processed` ou upload S3.
- `extractor.py`: gera `data/embeddings.csv` (local) ou CSV no S3.
- `load_embeddings.py`: ingestao no Postgres.
- `semantic_search_eval.py`: Issue 4 (Top-5/Top-10, L2 vs cosseno).
- `docker-compose.yml`: PostgreSQL + pgvector.
- `sql/semantic_search_queries.sql`: consultas pgvector.

## Execucao offline (Mac) — fechar Issue 4 agora

**Um comando por linha** (nao use `#` na mesma linha) ou rode:

```bash
bash run_offline_pipeline.sh
```

Manual:

```bash
python preprocess.py --mode local
python extractor.py --mode local
docker compose up -d
python load_embeddings.py --mode local
find data/processed -type f -iname '*.jpg' | head -3
python semantic_search_eval.py --image data/processed/PASTA/arquivo.jpg --query-class PASTA --top-k 5 10 --runs 5
```

Requer **Docker Desktop aberto**. Postgres do projeto usa **porta 5433** no Mac (5432 costuma estar ocupada por outros containers).

Saida da Issue 4: `docs/issue4_metric_evaluation.md`.

## Execucao na EC2 (AWS)

```bash
export PIPELINE_MODE=aws
python preprocess.py --mode aws
python extractor.py --mode aws
docker compose up -d
python load_embeddings.py --mode aws
python semantic_search_eval.py --image-id foto.jpg --top-k 5 10 --runs 5
```

## Documentacao

- [offline_and_aws.md](docs/offline_and_aws.md) — roteiro Mac vs VM
- [issue4_semantic_search.md](docs/issue4_semantic_search.md) — metricas e script
- [preprocess.md](docs/preprocess.md), [extractor.md](docs/extractor.md), [load_embeddings.md](docs/load_embeddings.md)
