import argparse
import cv2
import kagglehub
from pathlib import Path

from pipeline_config import (
    LOCAL_PROCESSED_DIR,
    TARGET_SIZE,
    DATASET_URI,
    S3_PREFIX,
    image_bucket,
    is_aws_mode,
    processed_image_key,
    upload_image_bytes,
    get_s3_client,
)


def download_dataset():
    print("Baixando dataset via Kagglehub...")
    return kagglehub.dataset_download(DATASET_URI)


def preprocess_to_disk(input_dir, output_dir, target_size):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Salvando imagens processadas em {output_path.resolve()}...")
    count_dict = {}

    for img_file in input_path.rglob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        class_name = img_file.parent.name
        if class_name == input_path.name:
            continue

        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            class_dir = output_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            out_file = class_dir / img_file.name
            cv2.imwrite(str(out_file), img_bgr_to_save)
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
        except Exception as exc:
            print(f"Erro ao processar {img_file.name}: {exc}")

    print("\nResumo do preprocessamento local:")
    for class_name, count in sorted(count_dict.items()):
        print(f" -> {count} imagens na classe '{class_name}'.")


def preprocess_and_upload(input_dir, target_size):
    """Preprocessa e grava todas as imagens no bucket iris-cv-latente-data."""
    s3_client = get_s3_client()
    input_path = Path(input_dir)
    bucket = image_bucket()
    print(f"Enviando imagens para s3://{bucket}/{S3_PREFIX}/...")
    count_dict = {}

    for img_file in input_path.rglob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        class_name = img_file.parent.name
        if class_name == input_path.name:
            continue

        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", img_bgr_to_save)

            s3_key = processed_image_key(class_name, img_file.name)
            upload_image_bytes(
                s3_key,
                buffer.tobytes(),
                content_type="image/jpeg",
                s3=s3_client,
            )
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
        except Exception as exc:
            print(f"Erro ao processar {img_file.name}: {exc}")

    print(f"\nResumo do upload para s3://{bucket}/{S3_PREFIX}/:")
    for class_name, count in sorted(count_dict.items()):
        print(f" -> {count} imagens enviadas para a classe '{class_name}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessamento Oxford Flower 17.")
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help=(
            "local: grava em data/processed | "
            "aws: upload no bucket iris-cv-latente-data."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_dataset_path = download_dataset()

    if args.mode == "local":
        preprocess_to_disk(raw_dataset_path, LOCAL_PROCESSED_DIR, TARGET_SIZE)
    else:
        preprocess_and_upload(raw_dataset_path, TARGET_SIZE)
