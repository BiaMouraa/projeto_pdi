import os
import tempfile
from dataclasses import dataclass
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
S3_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_SIZE = (256, 256)

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "iris_latente"),
    "user": os.getenv("PGUSER", "admin"),
    "password": os.getenv("PGPASSWORD", "adminpassword"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5433"),
}

# Pastas de split comuns no Oxford-102 (PyTorch Challenge).
SPLIT_DIR_NAMES = frozenset({"train", "valid", "validation", "val", "test"})


@dataclass(frozen=True)
class DatasetSpec:
    """Configuracao de um dataset do pipeline."""

    key: str
    slug: str
    kaggle_uri: str
    table_name: str
    description: str

    @property
    def local_root(self) -> Path:
        return PROJECT_ROOT / "data" / self.slug

    @property
    def local_processed_dir(self) -> Path:
        return self.local_root / "processed"

    @property
    def local_embeddings_csv(self) -> Path:
        return self.local_root / "embeddings.csv"

    @property
    def s3_processed_prefix(self) -> str:
        return f"{self.slug}/processed"

    @property
    def s3_embeddings_key(self) -> str:
        return f"{self.slug}/embeddings/embeddings.csv"

    @property
    def s3_reports_prefix(self) -> str:
        return f"{self.slug}/reports"

    def embeddings_csv_uri(self, mode=None) -> str:
        use_aws = (mode == "aws") if mode is not None else is_aws_mode()
        if use_aws:
            return f"s3://{image_bucket()}/{self.s3_embeddings_key}"
        return str(self.local_embeddings_csv)

    def processed_image_key(self, class_name, filename) -> str:
        return f"{self.s3_processed_prefix}/{class_name}/{filename}"


DATASETS = {
    "oxford17": DatasetSpec(
        key="oxford17",
        slug="oxford-flower-17",
        kaggle_uri="cantonioupao/oxford-flower-17categories-labelled",
        # Mantem a tabela legada para nao perder embeddings ja carregados.
        table_name="flower_embeddings",
        description="Oxford Flower 17 (desenvolvimento)",
    ),
    "oxford102": DatasetSpec(
        key="oxford102",
        slug="oxford-flower-102",
        kaggle_uri="nunenuh/pytorch-challange-flower-dataset",
        table_name="flower_embeddings_oxford102",
        description="Oxford Flower 102 (PyTorch Challenge)",
    ),
}

DATASET_CHOICES = tuple(DATASETS.keys())
DEFAULT_DATASET_KEY = os.getenv("DATASET", "oxford17").strip().lower()
ALLOWED_TABLES = frozenset(spec.table_name for spec in DATASETS.values())

# Compatibilidade com imports antigos.
DATASET_URI = DATASETS["oxford17"].kaggle_uri
LOCAL_PROCESSED_DIR = DATASETS["oxford17"].local_processed_dir
LOCAL_EMBEDDINGS_CSV = DATASETS["oxford17"].local_embeddings_csv
S3_PREFIX = DATASETS["oxford17"].s3_processed_prefix


def is_aws_mode():
    return PIPELINE_MODE == "aws"


def image_bucket():
    """Bucket obrigatorio para armazenamento de imagens no fluxo aws."""
    return AWS_IMAGE_BUCKET


def resolve_dataset(dataset_key=None) -> DatasetSpec:
    key = (dataset_key or DEFAULT_DATASET_KEY).strip().lower()
    if key not in DATASETS:
        raise ValueError(
            f"Dataset desconhecido: {key!r}. Opcoes: {', '.join(DATASET_CHOICES)}"
        )
    return DATASETS[key]


def add_dataset_argument(parser):
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET_KEY if DEFAULT_DATASET_KEY in DATASETS else "oxford17",
        help=(
            "oxford17: desenvolvimento (17 classes, tabela flower_embeddings). "
            "oxford102: 102 classes (tabela flower_embeddings_oxford102). "
            "No S3, artefatos ficam em oxford-flower-17/ ou oxford-flower-102/."
        ),
    )
    return parser


def embeddings_csv_source(dataset_key=None, mode=None):
    return resolve_dataset(dataset_key).embeddings_csv_uri(mode=mode)


def processed_image_key(class_name, filename, dataset_key=None):
    return resolve_dataset(dataset_key).processed_image_key(class_name, filename)


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


def list_image_keys(prefix=None, s3=None, dataset_key=None):
    """Lista chaves de imagem no bucket oficial sob o prefixo dado."""
    client = s3 or get_s3_client()
    if prefix is None:
        prefix = f"{resolve_dataset(dataset_key).s3_processed_prefix.rstrip('/')}/"
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=image_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".jpeg", ".png")):
                keys.append(key)
    return keys


def guess_content_type(path):
    return {
        ".md": "text/markdown; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
        ".txt": "text/plain; charset=utf-8",
    }.get(Path(path).suffix.lower(), "application/octet-stream")


def upload_artifact(local_path, s3_key, s3=None):
    """Envia um arquivo local (relatorio/grafico) para o bucket."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Arquivo para upload nao encontrado: {local_path}")

    client = s3 or get_s3_client()
    key = str(s3_key).lstrip("/")
    client.put_object(
        Bucket=image_bucket(),
        Key=key,
        Body=local_path.read_bytes(),
        ContentType=guess_content_type(local_path),
    )
    uri = image_s3_uri(key)
    print(f"Artefato enviado: {uri}")
    return uri


def upload_report_artifacts(paths, dataset_key=None, s3=None):
    """Envia relatorios/graficos para s3://bucket/<slug>/reports/...

    Em modo local nao faz nada. Preserva o caminho relativo a docs/ quando possivel.
    """
    if not is_aws_mode():
        return []

    dataset = resolve_dataset(dataset_key)
    prefix = dataset.s3_reports_prefix
    docs_root = (PROJECT_ROOT / "docs").resolve()
    client = s3 or get_s3_client()
    uploaded = []

    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(docs_root).as_posix()
        except ValueError:
            rel = path.name
        s3_key = f"{prefix}/{rel}"
        uploaded.append(upload_artifact(resolved, s3_key, s3=client))

    if uploaded:
        print(
            f"{len(uploaded)} artefato(s) em "
            f"s3://{image_bucket()}/{prefix}/"
        )
    return uploaded
