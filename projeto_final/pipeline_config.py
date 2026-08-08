import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# local: Mac/dev offline (sem credenciais AWS)
# aws: EC2 com IAM role (S3)
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "local").strip().lower()

# Bucket unico para armazenamento de imagens (e artefatos) no modo aws.
AWS_IMAGE_BUCKET = "iris-cv-latente-data"
BUCKET_NAME = AWS_IMAGE_BUCKET if PIPELINE_MODE == "aws" else os.getenv(
    "S3_BUCKET", AWS_IMAGE_BUCKET
)
S3_PREFIX = os.getenv("S3_PREFIX", "processed").strip("/")
S3_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

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


def image_bucket():
    """Bucket obrigatorio para armazenamento de imagens no fluxo aws."""
    return AWS_IMAGE_BUCKET


def embeddings_csv_source():
    if is_aws_mode():
        return f"s3://{image_bucket()}/embeddings/embeddings.csv"
    return str(LOCAL_EMBEDDINGS_CSV)


def processed_image_key(class_name, filename):
    return f"{S3_PREFIX}/{class_name}/{filename}"


def image_s3_uri(key):
    return f"s3://{image_bucket()}/{str(key).lstrip('/')}"


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"URI S3 invalida: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def normalize_image_s3_uri(uri):
    """Reescreve URI s3:// para o bucket oficial de imagens do projeto."""
    _bucket, key = parse_s3_uri(uri)
    return image_s3_uri(key)


def get_s3_client(region_name=None):
    import boto3

    return boto3.client("s3", region_name=region_name or S3_REGION)


def upload_image_bytes(key, body, content_type="image/jpeg", s3=None):
    """Grava bytes de imagem no bucket iris-cv-latente-data."""
    client = s3 or get_s3_client()
    key = str(key).lstrip("/")
    client.put_object(
        Bucket=image_bucket(),
        Key=key,
        Body=body,
        ContentType=content_type,
    )
    return image_s3_uri(key)


def download_image_bytes(uri_or_key, s3=None):
    """Baixa bytes de imagem do bucket oficial (aceita s3:// ou chave)."""
    client = s3 or get_s3_client()
    raw = str(uri_or_key)
    if raw.startswith("s3://"):
        bucket, key = parse_s3_uri(raw)
        # Em modo aws, forca o bucket do projeto mesmo se a URI apontar outro.
        if is_aws_mode() or bucket == AWS_IMAGE_BUCKET:
            bucket = image_bucket()
    else:
        bucket, key = image_bucket(), raw.lstrip("/")

    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read(), key


def download_image_to_temp(uri_or_key, s3=None):
    data, key = download_image_bytes(uri_or_key, s3=s3)
    suffix = Path(key).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def list_image_keys(prefix=None, s3=None):
    """Lista chaves de imagem no bucket oficial sob o prefixo dado."""
    client = s3 or get_s3_client()
    prefix = (prefix if prefix is not None else f"{S3_PREFIX.rstrip('/')}/")
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=image_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".jpeg", ".png")):
                keys.append(key)
    return keys
