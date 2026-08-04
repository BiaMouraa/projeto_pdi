import pandas as pd
import psycopg2
from tqdm import tqdm

DB_CONFIG = {
    "dbname": "iris_latente",
    "user": "admin",
    "password": "adminpassword",
    "host": "localhost",
    "port": "5432"
}

# A URI gerada pela fase de extração no S3
CSV_S3_URI = "s3://iris-cv-latente-data/embeddings/embeddings.csv"

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

SQL_INDEX = "CREATE INDEX ON flower_embeddings USING hnsw (embedding vector_cosine_ops);"
SQL_INSERT = "INSERT INTO flower_embeddings (image_id, class_name, file_path, embedding) VALUES (%s, %s, %s, %s)"

def carregar_dados_no_banco():
    print(f"Lendo dados diretamente do S3: {CSV_S3_URI}...")
    # Novamente, o s3fs trabalha em background para carregar os dados
    df = pd.read_csv(CSV_S3_URI)
    
    print("Conectando ao PostgreSQL via Docker...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        print("Configurando esquema e tabela...")
        cursor.execute(SQL_SETUP)
        
        print("Iniciando ingestão de registros no BD...")
        # Uso do bloco transacional
        for _, row in tqdm(df.iterrows(), total=len(df)):
            cursor.execute(SQL_INSERT, (
                row['image_id'],
                row['class_name'],
                row['file_path'],  # Agora armazena a URI completa do S3
                row['embedding']
            ))
            
        conn.commit()
        print("Processo de inserção finalizado!")

        print("Construindo índice vetorial HNSW...")
        cursor.execute(SQL_INDEX)
        conn.commit()
        print("Infraestrutura de dados completa e otimizada.")

    except Exception as e:
        print(f"ROLLBACK. Ocorreu um erro na ingestão de dados: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    carregar_dados_no_banco()