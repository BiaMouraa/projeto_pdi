import argparse
import io
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from pipeline_config import (
    LOCAL_EMBEDDINGS_CSV,
    LOCAL_PROCESSED_DIR,
    S3_PREFIX,
    download_image_bytes,
    get_s3_client,
    image_bucket,
    image_s3_uri,
    is_aws_mode,
    list_image_keys,
)

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_feature_extractor():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    return model


def extract_embeddings_local(processed_dir, output_csv):
    processed_path = processed_dir.resolve()
    if not processed_path.is_dir():
        raise FileNotFoundError(
            f"Pasta nao encontrada: {processed_path}. Rode: python preprocess.py --mode local"
        )

    class ImageFolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target, path

    dataset = ImageFolderWithPaths(str(processed_path), transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    model = get_feature_extractor()
    all_embeddings = []

    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader, desc="Extraindo embeddings (local)"):
            inputs = inputs.to(device)
            features = model(inputs).cpu().numpy()
            for i in range(len(features)):
                file_path = str(Path(paths[i]).resolve())
                all_embeddings.append(
                    {
                        "image_id": os.path.basename(file_path),
                        "class_name": idx_to_class[labels[i].item()],
                        "embedding": str(features[i].tolist()),
                        "file_path": file_path,
                    }
                )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_embeddings).to_csv(output_csv, index=False)
    print(f"Extracao concluida! CSV em {output_csv.resolve()}")


def extract_embeddings_aws():
    """Le imagens do bucket iris-cv-latente-data e grava CSV no mesmo bucket."""
    s3_client = get_s3_client()
    prefix = f"{S3_PREFIX.rstrip('/')}/"
    output_uri = f"s3://{image_bucket()}/embeddings/embeddings.csv"
    bucket = image_bucket()

    class S3FlowerDataset(Dataset):
        def __init__(self, s3_prefix, image_transform=None):
            self.transform = image_transform
            self.image_keys = list_image_keys(prefix=s3_prefix, s3=s3_client)
            self.classes = sorted(
                {key.split("/")[-2] for key in self.image_keys if "/" in key}
            )
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        def __len__(self):
            return len(self.image_keys)

        def __getitem__(self, idx):
            key = self.image_keys[idx]
            class_name = key.split("/")[-2]
            label = self.class_to_idx[class_name]
            data, _ = download_image_bytes(key, s3=s3_client)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, os.path.basename(key), image_s3_uri(key)

    print(f"Lendo imagens de s3://{bucket}/{prefix}")
    dataset = S3FlowerDataset(prefix, transform)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"Nenhuma imagem em s3://{bucket}/{prefix}. "
            "Rode: python preprocess.py --mode aws"
        )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    model = get_feature_extractor()
    all_embeddings = []

    with torch.no_grad():
        for inputs, labels, file_names, uris in tqdm(
            dataloader, desc=f"Extraindo embeddings (s3://{bucket})"
        ):
            inputs = inputs.to(device)
            features = model(inputs).cpu().numpy()
            for i in range(len(features)):
                all_embeddings.append(
                    {
                        "image_id": file_names[i],
                        "class_name": idx_to_class[labels[i].item()],
                        "embedding": str(features[i].tolist()),
                        "file_path": uris[i],
                    }
                )

    pd.DataFrame(all_embeddings).to_csv(output_uri, index=False)
    print(f"Extracao concluida! CSV em {output_uri}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extracao de embeddings MobileNetV2.")
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help=(
            "local: data/processed -> data/embeddings.csv | "
            "aws: iris-cv-latente-data -> S3."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default=str(LOCAL_PROCESSED_DIR),
        help="Entrada local (modo local).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(LOCAL_EMBEDDINGS_CSV),
        help="Saida CSV (modo local).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "local":
        extract_embeddings_local(Path(args.processed_dir), Path(args.output_csv))
    else:
        extract_embeddings_aws()
