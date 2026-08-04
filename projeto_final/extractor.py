import boto3
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
from tqdm import tqdm
import os

# ==========================================
# Configurações Iniciais
# ==========================================
BUCKET_NAME = "NOME_DO_SEU_BUCKET"
S3_PREFIX = "processed/"
OUTPUT_CSV_URI = f"s3://{BUCKET_NAME}/embeddings/embeddings.csv"

BATCH_SIZE = 32 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s3_client = boto3.client('s3', region_name='us-east-1')

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# Dataset Customizado para Leitura do S3
# ==========================================
class S3FlowerDataset(Dataset):
    def __init__(self, bucket, prefix, transform=None):
        self.bucket = bucket
        self.transform = transform
        self.image_keys = []
        self.classes = []
        
        print("Mapeando arquivos no S3...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_keys.append(key)
                    
                    # Extrai a classe da URI: processed/class_name/file.jpg
                    class_name = key.split('/')[-2]
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                        
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"{len(self.image_keys)} imagens mapeadas em {len(self.classes)} classes.")

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        key = self.image_keys[idx]
        class_name = key.split('/')[-2]
        label = self.class_to_idx[class_name]
        
        # Lê os bytes diretamente do S3
        response = s3_client.get_object(Bucket=self.bucket, Key=key)
        img_bytes = response['Body'].read()
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Retorna tensor, label numérico, nome do arquivo e URI completa
        return img, label, os.path.basename(key), f"s3://{self.bucket}/{key}"

def get_feature_extractor():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    return model

def extract_embeddings():
    dataset = S3FlowerDataset(bucket=BUCKET_NAME, prefix=S3_PREFIX, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    model = get_feature_extractor()
    all_embeddings = []
    
    with torch.no_grad():
        for inputs, labels, file_names, uris in tqdm(dataloader, desc="Extraindo Batch S3"):
            inputs = inputs.to(device)
            features = model(inputs).cpu().numpy()
            
            for i in range(len(features)):
                all_embeddings.append({
                    "image_id": file_names[i],
                    "class_name": idx_to_class[labels[i].item()],
                    "embedding": str(features[i].tolist()),
                    "file_path": uris[i]
                })
                
    print("\nSalvando embeddings estruturados no S3...")
    df = pd.DataFrame(all_embeddings)
    
    # O Pandas, com a biblioteca s3fs, escreve direto no bucket
    df.to_csv(OUTPUT_CSV_URI, index=False)
    print(f"Extração concluída! Dados salvos em {OUTPUT_CSV_URI}")

if __name__ == "__main__":
    extract_embeddings()