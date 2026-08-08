import argparse
from pathlib import Path

import pandas as pd
import psycopg2
from tqdm import tqdm

from pipeline_config import (
    DB_CONFIG,
    LOCAL_EMBEDDINGS_CSV,
    embeddings_csv_source,
    image_bucket,
    is_aws_mode,
)

SQL_SETUP = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS flower_embeddings;

CREATE TABLE flower_embeddings (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255),
    class_name VARCHAR(100),
    file_path TEXT,
    embedding VECTOR(1280)
);
"""

SQL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS flower_embeddings_embedding_cosine_idx ON flower_embeddings USING hnsw (embedding vector_cosine_ops);",
    "CREATE INDEX IF NOT EXISTS flower_embeddings_embedding_l2_idx ON flower_embeddings USING hnsw (embedding vector_l2_ops);",
]
SQL_INSERT = (
    "INSERT INTO flower_embeddings (image_id, class_name, file_path, embedding) "
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
        "Rode extractor.py --mode local ou informe --csv."
    )


def carregar_dados_no_banco(csv_path):
    df = read_embeddings_csv(csv_path)

    print("Conectando ao PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        print("Configurando esquema e tabela...")
        cursor.execute(SQL_SETUP)

        print("Iniciando ingestao de registros no BD...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            cursor.execute(
                SQL_INSERT,
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
        for sql_index in SQL_INDEXES:
            cursor.execute(sql_index)
        conn.commit()
        print("Infraestrutura pronta (cosseno + euclidiana).")

    except Exception as exc:
        print(f"ROLLBACK. Erro na ingestao: {exc}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def parse_args():
    default_csv = embeddings_csv_source()
    parser = argparse.ArgumentParser(description="Ingestao de embeddings no PostgreSQL.")
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=f"CSV de embeddings (padrao: {default_csv}).",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help=(
            "Atalho para origem do CSV. "
            f"aws: s3://{image_bucket()}/embeddings/embeddings.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_local = str(LOCAL_EMBEDDINGS_CSV)
    default_aws = f"s3://{image_bucket()}/embeddings/embeddings.csv"

    if args.csv in {embeddings_csv_source(), default_local, default_aws}:
        csv_path = default_local if args.mode == "local" else default_aws
    else:
        csv_path = args.csv

    carregar_dados_no_banco(csv_path)
