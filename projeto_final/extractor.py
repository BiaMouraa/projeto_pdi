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
    add_dataset_argument,
    download_image_bytes,
    get_s3_client,
    image_bucket,
    image_s3_uri,
    is_aws_mode,
    list_image_keys,
    resolve_dataset,
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
            f"Pasta nao encontrada: {processed_path}. "
            "Rode: python preprocess.py --mode local --dataset ..."
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


def extract_embeddings_aws(dataset_spec):
    """Le imagens do subdiretorio do dataset no bucket e grava CSV no mesmo slug."""
    s3_client = get_s3_client()
    prefix = f"{dataset_spec.s3_processed_prefix.rstrip('/')}/"
    output_uri = dataset_spec.embeddings_csv_uri(mode="aws")
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
    flower_dataset = S3FlowerDataset(prefix, transform)
    if len(flower_dataset) == 0:
        raise FileNotFoundError(
            f"Nenhuma imagem em s3://{bucket}/{prefix}. "
            f"Rode: python preprocess.py --mode aws --dataset {dataset_spec.key}"
        )

    dataloader = DataLoader(flower_dataset, batch_size=BATCH_SIZE, shuffle=False)
    idx_to_class = {idx: name for name, idx in flower_dataset.class_to_idx.items()}
    model = get_feature_extractor()
    all_embeddings = []

    with torch.no_grad():
        for inputs, labels, file_names, uris in tqdm(
            dataloader, desc=f"Extraindo embeddings (s3://{bucket}/{dataset_spec.slug})"
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
    add_dataset_argument(parser)
    parser.add_argument(
        "--mode",
        choices=("local", "aws"),
        default="local" if not is_aws_mode() else "aws",
        help=(
            "local: data/<slug>/processed -> data/<slug>/embeddings.csv | "
            "aws: s3://.../<slug>/processed -> s3://.../<slug>/embeddings/."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Entrada local (padrao: data/<slug>/processed).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Saida CSV local (padrao: data/<slug>/embeddings.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_spec = resolve_dataset(args.dataset)
    print(f"Dataset: {dataset_spec.key} ({dataset_spec.description})")

    if args.mode == "local":
        processed_dir = Path(args.processed_dir or dataset_spec.local_processed_dir)
        output_csv = Path(args.output_csv or dataset_spec.local_embeddings_csv)
        extract_embeddings_local(processed_dir, output_csv)
    else:
        extract_embeddings_aws(dataset_spec)
