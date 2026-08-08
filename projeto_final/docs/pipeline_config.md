# Documentacao do pipeline_config.py

Descreve [`pipeline_config.py`](../pipeline_config.py), o modulo central de
configuracao compartilhado por todos os scripts do `projeto_final`.

## 1. Objetivo

Centralizar em um so lugar:

- o **modo** de execucao (local vs AWS);
- os caminhos locais (imagens processadas, CSV de embeddings);
- os dados de conexao com o PostgreSQL;
- a origem do CSV de embeddings conforme o modo.

Assim, o mesmo codigo roda offline no Mac e na EC2 sem edicoes manuais.

## 2. Variaveis principais

| Variavel | Descricao | Como sobrescrever |
|----------|-----------|-------------------|
| `PIPELINE_MODE` | `local` (padrao) ou `aws` | env `PIPELINE_MODE` |
| `BUCKET_NAME` | bucket S3 (modo aws) | env `S3_BUCKET` |
| `S3_PREFIX` | prefixo das imagens no S3 | env `S3_PREFIX` |
| `LOCAL_PROCESSED_DIR` | `data/processed` | fixo no projeto |
| `LOCAL_EMBEDDINGS_CSV` | `data/embeddings.csv` | fixo no projeto |
| `DB_CONFIG` | host/porta/credenciais do Postgres | env `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` |

O `DB_CONFIG` usa por padrao a **porta 5433** (o `docker-compose.yml` mapeia
`5433:5432` para evitar conflito com outro Postgres local na 5432).

## 3. Funcoes utilitarias

- `is_aws_mode()`: retorna `True` se `PIPELINE_MODE == "aws"`.
- `embeddings_csv_source()`: devolve o caminho do CSV de embeddings
  (`data/embeddings.csv` no local, `s3://.../embeddings.csv` no aws).

## 4. Uso pelos scripts

- `preprocess.py`, `extractor.py`: escolhem local vs S3 a partir do modo.
- `load_embeddings.py`: le o CSV da fonte correta e conecta com `DB_CONFIG`.
- `semantic_search_eval.py`, `retrieval_evaluation.py`: usam `DB_CONFIG` como
  padrao de conexao (com opcao de sobrescrever via `--db-*`).

## 5. Exemplo

```bash
# Offline (padrao)
python extractor.py --mode local

# Na EC2
export PIPELINE_MODE=aws
python extractor.py --mode aws
```

Veja tambem [offline_and_aws.md](offline_and_aws.md).
