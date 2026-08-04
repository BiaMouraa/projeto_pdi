import cv2
import kagglehub
import boto3
from pathlib import Path

# ==========================================
# Configurações Iniciais
# ==========================================
BUCKET_NAME = "iris-cv-latente-data"
S3_PREFIX = "processed"
DATASET_URI = "cantonioupao/oxford-flower-17categories-labelled"
TARGET_SIZE = (256, 256)

# O boto3 assume as permissões da IAM Role da EC2 automaticamente
s3_client = boto3.client('s3', region_name='us-east-1')

def download_dataset():
    print("Baixando dataset no cache da EC2 via Kagglehub...")
    path = kagglehub.dataset_download(DATASET_URI)
    return path

def preprocess_and_upload(input_dir, target_size):
    input_path = Path(input_dir)
    print(f"Iniciando processamento e upload para s3://{BUCKET_NAME}/{S3_PREFIX}/...")
    
    count_dict = {}

    for img_file in input_path.rglob("*.*"):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        class_name = img_file.parent.name
        if class_name == input_path.name:
            continue
            
        try:
            # 1. Carrega e trata a imagem
            img = cv2.imread(str(img_file))
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            
            # 2. Codifica a imagem em memória (buffer) em vez de salvar no disco
            _, buffer = cv2.imencode('.jpg', img_bgr_to_save)
            
            # 3. Define o caminho no S3 (URI) e faz o upload dos bytes
            s3_key = f"{S3_PREFIX}/{class_name}/{img_file.name}"
            
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )
            
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
            
        except Exception as e:
            print(f"Erro ao processar {img_file.name}: {e}")

    print("\nResumo do Upload para o S3:")
    for c_name, c_count in sorted(count_dict.items()):
        print(f" -> {c_count} imagens enviadas para a classe '{c_name}'.")

if __name__ == "__main__":
    raw_dataset_path = download_dataset()
    preprocess_and_upload(raw_dataset_path, TARGET_SIZE)