import argparse
import re
from pathlib import Path

import pandas as pd
import psycopg2
from tqdm import tqdm

from pipeline_config import (
    ALLOWED_TABLES,
    DB_CONFIG,
    add_dataset_argument,
    is_aws_mode,
    resolve_dataset,
)


def assert_safe_table(table_name):
    if table_name not in ALLOWED_TABLES or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", table_name):
        raise ValueError(f"Nome de tabela nao permitido: {table_name!r}")
    return table_name


def sql_setup(table_name):
    table = assert_safe_table(table_name)
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255),
    class_name VARCHAR(100),
    file_path TEXT,
    embedding VECTOR(1280)
);

TRUNCATE TABLE {table} RESTART IDENTITY;
"""


def sql_indexes(table_name):
    table = assert_safe_table(table_name)
    return [
        (
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_cosine_idx "
            f"ON {table} USING hnsw (embedding vector_cosine_ops);"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_l2_idx "
            f"ON {table} USING hnsw (embedding vector_l2_ops);"
        ),
    ]


def sql_insert(table_name):
    table = assert_safe_table(table_name)
    return (
        f"INSERT INTO {table} (image_id, class_name, file_path, embedding) "
        "VALUES (%s, %s, %s, %s)"
    )


def read_embeddings_csv(csv_path):
    path = Path(csv_path)
    if path.is_file():
        print(f"Lendo CSV local: {path.resolve()}")
        return pd.read_csv(path)

    if str(csv_path).startswith("s3://"):
        print(f"Lendo CSV do S3: {csv_path}")
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"CSV nao encontrado: {csv_path}. "
        "Rode extractor.py --mode local/--mode aws --dataset ... ou informe --csv."
    )


def carregar_dados_no_banco(csv_path, table_name):
    """Recarrega apenas a tabela do dataset selecionado (nao apaga as demais)."""
    table = assert_safe_table(table_name)
    df = read_embeddings_csv(csv_path)

    print("Conectando ao PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        print(f"Configurando tabela {table} (outras tabelas de embeddings sao preservadas)...")
        cursor.execute(sql_setup(table))

        insert_sql = sql_insert(table)
        print(f"Iniciando ingestao de {len(df)} registros em {table}...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            cursor.execute(
                insert_sql,
                (
                    row["image_id"],
                    row["class_name"],
                    row["file_path"],
                    row["embedding"],
                ),
            )

        conn.commit()
        print("Processo de insercao finalizado!")

        print("Construindo indices vetoriais HNSW...")
        for sql_index in sql_indexes(table):
            cursor.execute(sql_index)
        conn.commit()
        print(f"Infraestrutura pronta em {table} (cosseno + euclidiana).")

    except Exception as exc:
        print(f"ROLLBACK. Erro na ingestao: {exc}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Ingestao de embeddings no PostgreSQL.")
    add_dataset_argument(parser)
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV de embeddings (padrao depende de --dataset e --mode).",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help="Atalho para origem padrao do CSV (local vs S3 no slug do dataset).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    default_csv = dataset.embeddings_csv_uri(mode=args.mode)
    csv_path = args.csv or default_csv

    print(f"Dataset: {dataset.key} ({dataset.description})")
    print(f"Tabela destino: {dataset.table_name}")
    print(f"CSV: {csv_path}")

    carregar_dados_no_banco(csv_path, dataset.table_name)
