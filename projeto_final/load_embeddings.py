import pandas as pd
import psycopg2
from tqdm import tqdm

# ==========================================
# 1. Configurações de Conexão (Baseado no Docker)
# ==========================================
DB_CONFIG = {
    "dbname": "iris_latente",
    "user": "admin",
    "password": "adminpassword",
    "host": "localhost",
    "port": "5432"
}

CSV_FILE = "data/embeddings.csv"

# ==========================================
# 2. Queries SQL
# ==========================================
SQL_SETUP = """
-- Habilita a extensão
CREATE EXTENSION IF NOT EXISTS vector;

-- Remove a tabela se existir (para facilitar testes locais)
DROP TABLE IF EXISTS flower_embeddings;

-- Criação da tabela. A MobileNetV2 gera vetores de tamanho 1280.
CREATE TABLE flower_embeddings (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255),
    class_name VARCHAR(100),
    file_path TEXT,
    embedding VECTOR(1280)
);
"""

# HNSW é o índice mais rápido atualmente para busca aproximada no pgvector
SQL_INDEX = """
CREATE INDEX ON flower_embeddings 
USING hnsw (embedding vector_cosine_ops);
"""

SQL_INSERT = """
INSERT INTO flower_embeddings (image_id, class_name, file_path, embedding)
VALUES (%s, %s, %s, %s)
"""

def carregar_dados_no_banco():
    print("Conectando ao PostgreSQL via Docker...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # 1. Setup da extensão e tabela
        print("Configurando extensão e modelando tabela...")
        cursor.execute(SQL_SETUP)
        conn.commit()

        # 2. Leitura dos dados extraídos
        print(f"Lendo dados do arquivo {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        
        # 3. Inserção iterativa
        print("Iniciando ingestão de dados...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            cursor.execute(SQL_INSERT, (
                row['image_id'],
                row['class_name'],
                row['file_path'],
                row['embedding']
            ))
        
        conn.commit()
        print("Dados ingeridos com sucesso!")

        # 4. Criação do Índice
        print("Construindo índice HNSW para otimização de busca (Isso pode levar alguns segundos)...")
        cursor.execute(SQL_INDEX)
        conn.commit()
        print("Índice criado com sucesso!")

    except Exception as e:
        print(f"Ocorreu um erro no banco de dados: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        print("Conexão encerrada.")

if __name__ == "__main__":
    carregar_dados_no_banco()