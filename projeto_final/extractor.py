import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from pipeline_config import (
    BUCKET_NAME,
    LOCAL_EMBEDDINGS_CSV,
    LOCAL_PROCESSED_DIR,
    S3_PREFIX,
    embeddings_csv_source,
    is_aws_mode,
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
    import boto3
    import io
    from torch.utils.data import Dataset

    s3_client = boto3.client("s3", region_name="us-east-1")
    prefix = f"{S3_PREFIX.rstrip('/')}/"
    output_uri = embeddings_csv_source()

    class S3FlowerDataset(Dataset):
        def __init__(self, bucket, s3_prefix, image_transform=None):
            self.bucket = bucket
            self.transform = image_transform
            self.image_keys = []
            self.classes = []

            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_keys.append(key)
                        class_name = key.split("/")[-2]
                        if class_name not in self.classes:
                            self.classes.append(class_name)

            self.classes.sort()
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        def __len__(self):
            return len(self.image_keys)

        def __getitem__(self, idx):
            key = self.image_keys[idx]
            class_name = key.split("/")[-2]
            label = self.class_to_idx[class_name]
            response = s3_client.get_object(Bucket=self.bucket, Key=key)
            img = Image.open(io.BytesIO(response["Body"].read())).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, os.path.basename(key), f"s3://{self.bucket}/{key}"

    dataset = S3FlowerDataset(BUCKET_NAME, prefix, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    model = get_feature_extractor()
    all_embeddings = []

    with torch.no_grad():
        for inputs, labels, file_names, uris in tqdm(dataloader, desc="Extraindo embeddings (S3)"):
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
        help="local: data/processed -> data/embeddings.csv | aws: S3 -> S3.",
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
