import argparse
import cv2
import kagglehub
from pathlib import Path

from pipeline_config import (
    BUCKET_NAME,
    DATASET_URI,
    LOCAL_PROCESSED_DIR,
    S3_PREFIX,
    TARGET_SIZE,
    is_aws_mode,
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
    import boto3

    s3_client = boto3.client("s3", region_name="us-east-1")
    input_path = Path(input_dir)
    print(f"Enviando para s3://{BUCKET_NAME}/{S3_PREFIX}/...")
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

            s3_key = f"{S3_PREFIX}/{class_name}/{img_file.name}"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=buffer.tobytes(),
                ContentType="image/jpeg",
            )
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
        except Exception as exc:
            print(f"Erro ao processar {img_file.name}: {exc}")

    print("\nResumo do upload para o S3:")
    for class_name, count in sorted(count_dict.items()):
        print(f" -> {count} imagens enviadas para a classe '{class_name}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessamento Oxford Flower 17.")
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help="local: grava em data/processed | aws: upload S3 (EC2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_dataset_path = download_dataset()

    if args.mode == "local":
        preprocess_to_disk(raw_dataset_path, LOCAL_PROCESSED_DIR, TARGET_SIZE)
    else:
        preprocess_and_upload(raw_dataset_path, TARGET_SIZE)
