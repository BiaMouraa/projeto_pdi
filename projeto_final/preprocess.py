import argparse
import json
import re
from pathlib import Path

import cv2
import kagglehub

from pipeline_config import (
    SPLIT_DIR_NAMES,
    TARGET_SIZE,
    add_dataset_argument,
    get_s3_client,
    image_bucket,
    is_aws_mode,
    resolve_dataset,
    upload_image_bytes,
)


def sanitize_class_name(name):
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(name)).strip()
    return cleaned or "unknown"


def download_dataset(kaggle_uri):
    print(f"Baixando dataset via Kagglehub: {kaggle_uri}")
    return kagglehub.dataset_download(kaggle_uri)


def load_class_name_map(dataset_root):
    """Carrega cat_to_name.json do Oxford-102, se existir."""
    root = Path(dataset_root)
    for candidate in root.rglob("cat_to_name.json"):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            mapping = {str(k): str(v) for k, v in data.items()}
            print(f"Mapeamento de classes carregado de {candidate}")
            return mapping
        except Exception as exc:
            print(f"Aviso: falha ao ler {candidate}: {exc}")
    return {}


def resolve_class_and_filename(img_file, dataset_root, class_name_map):
    """Extrai classe e nome de saida unico (trata train/valid/test do Oxford-102)."""
    img_file = Path(img_file)
    dataset_root = Path(dataset_root)
    folder_name = img_file.parent.name

    if folder_name == dataset_root.name or folder_name.lower() in SPLIT_DIR_NAMES:
        return None, None

    class_name = class_name_map.get(folder_name, folder_name)
    # Tambem tenta chave numerica sem zeros a esquerda (ex.: "01" -> "1").
    if class_name == folder_name and folder_name.isdigit():
        class_name = class_name_map.get(str(int(folder_name)), folder_name)
    class_name = sanitize_class_name(class_name)

    grandparent = img_file.parent.parent.name if img_file.parent.parent else ""
    if grandparent.lower() in SPLIT_DIR_NAMES:
        out_name = f"{grandparent}_{img_file.name}"
    else:
        out_name = img_file.name

    return class_name, out_name


def iter_dataset_images(input_dir, class_name_map=None):
    input_path = Path(input_dir)
    class_name_map = class_name_map or {}
    for img_file in input_path.rglob("*.*"):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        class_name, out_name = resolve_class_and_filename(
            img_file, input_path, class_name_map
        )
        if not class_name or not out_name:
            continue
        yield img_file, class_name, out_name


def preprocess_to_disk(input_dir, output_dir, target_size, class_name_map=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Salvando imagens processadas em {output_path.resolve()}...")
    count_dict = {}

    for img_file, class_name, out_name in iter_dataset_images(input_dir, class_name_map):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            class_dir = output_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            out_file = class_dir / out_name
            cv2.imwrite(str(out_file), img_bgr_to_save)
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
        except Exception as exc:
            print(f"Erro ao processar {img_file.name}: {exc}")

    print("\nResumo do preprocessamento local:")
    for class_name, count in sorted(count_dict.items()):
        print(f" -> {count} imagens na classe '{class_name}'.")
    print(f"Total de classes: {len(count_dict)}")


def preprocess_and_upload(input_dir, target_size, dataset, class_name_map=None):
    """Preprocessa e grava imagens em s3://bucket/<slug>/processed/."""
    s3_client = get_s3_client()
    bucket = image_bucket()
    prefix = dataset.s3_processed_prefix
    print(f"Enviando imagens para s3://{bucket}/{prefix}/...")
    count_dict = {}

    for img_file, class_name, out_name in iter_dataset_images(input_dir, class_name_map):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", img_bgr_to_save)

            s3_key = dataset.processed_image_key(class_name, out_name)
            upload_image_bytes(
                s3_key,
                buffer.tobytes(),
                content_type="image/jpeg",
                s3=s3_client,
            )
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
        except Exception as exc:
            print(f"Erro ao processar {img_file.name}: {exc}")

    print(f"\nResumo do upload para s3://{bucket}/{prefix}/:")
    for class_name, count in sorted(count_dict.items()):
        print(f" -> {count} imagens enviadas para a classe '{class_name}'.")
    print(f"Total de classes: {len(count_dict)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessamento Oxford Flower 17 / 102."
    )
    add_dataset_argument(parser)
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help=(
            "local: grava em data/<slug>/processed | "
            "aws: upload no bucket iris-cv-latente-data/<slug>/."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    print(f"Dataset: {dataset.key} ({dataset.description})")
    print(f"Slug S3/local: {dataset.slug}")

    raw_dataset_path = download_dataset(dataset.kaggle_uri)
    class_name_map = load_class_name_map(raw_dataset_path)

    if args.mode == "local":
        preprocess_to_disk(
            raw_dataset_path,
            dataset.local_processed_dir,
            TARGET_SIZE,
            class_name_map=class_name_map,
        )
    else:
        preprocess_and_upload(
            raw_dataset_path,
            TARGET_SIZE,
            dataset,
            class_name_map=class_name_map,
        )
