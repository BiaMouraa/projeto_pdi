import os
from pathlib import Path

# local: Mac/dev offline (sem credenciais AWS)
# aws: EC2 com IAM role (S3)
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "local").strip().lower()

BUCKET_NAME = os.getenv("S3_BUCKET", "iris-cv-latente-data")
S3_PREFIX = os.getenv("S3_PREFIX", "processed").strip("/")

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOCAL_EMBEDDINGS_CSV = PROJECT_ROOT / "data" / "embeddings.csv"

DATASET_URI = "cantonioupao/oxford-flower-17categories-labelled"
TARGET_SIZE = (256, 256)

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "iris_latente"),
    "user": os.getenv("PGUSER", "admin"),
    "password": os.getenv("PGPASSWORD", "adminpassword"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5433"),
}


def is_aws_mode():
    return PIPELINE_MODE == "aws"


def embeddings_csv_source():
    if is_aws_mode():
        return f"s3://{BUCKET_NAME}/embeddings/embeddings.csv"
    return str(LOCAL_EMBEDDINGS_CSV)
