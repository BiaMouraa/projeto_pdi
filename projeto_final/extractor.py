import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# 1. Configurações Iniciais
# ==========================================
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = "data/embeddings.csv"

# Tamanho do lote (batch) para não estourar a memória (ajuste conforme a máquina)
BATCH_SIZE = 32 

# Configuração de dispositivo (Usa GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando inferência em: {device}")

# ==========================================
# 2. Pipeline de Transformação (Normalização)
# ==========================================
# A MobileNetV2 espera imagens 224x224 e normalizadas pelos padrões do ImageNet
transform = transforms.Compose([
    transforms.CenterCrop(224), # Nossas imagens estão 256x256, pegamos o centro
    transforms.ToTensor(),      # Converte para Tensor (já faz a divisão por 255 internamente)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Carregamento do Modelo (MobileNet V2)
# ==========================================
def get_feature_extractor():
    """Carrega a MobileNetV2 e remove a camada de classificação final."""
    # Carrega a rede pré-treinada
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # A MobileNetV2 no PyTorch guarda a classificação no bloco 'classifier'.
    # Substituímos esse bloco inteiro por uma camada de Identidade (pass-through).
    # Assim, a saída será o vetor de características bruto (1280 dimensões).
    model.classifier = nn.Identity()
    
    model.to(device)
    model.eval() # Modo de avaliação (desliga Dropout/BatchNorm updates)
    return model

# ==========================================
# 4. Processamento e Extração
# ==========================================
def extract_embeddings():
    print("Carregando o dataset local...")
    
    # O ImageFolder lê a estrutura de pastas e já mapeia o nome da pasta como 'classe'
    dataset = datasets.ImageFolder(root=PROCESSED_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Mapeamento do índice numérico gerado pelo PyTorch de volta para o nome da classe
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    model = get_feature_extractor()
    
    all_embeddings = []
    
    print("Iniciando extração de embeddings no espaço latente...")
    # Desativa cálculo de gradientes (economiza MUITA memória)
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Processando Batches"):
            inputs = inputs.to(device)
            
            # Passa pela rede
            features = model(inputs)
            
            # Move os vetores de volta para a CPU e converte para NumPy
            features = features.cpu().numpy()
            
            for i in range(len(features)):
                # Recuperar o caminho original do arquivo (para usar como ID/URI)
                # dataloader.dataset.samples guarda uma tupla (caminho_arquivo, label_idx)
                # Como não demos shuffle, a ordem sequencial é mantida
                global_idx = len(all_embeddings)
                file_path, _ = dataset.samples[global_idx]
                file_name = os.path.basename(file_path)
                
                # Estruturação voltada para inserção no banco relacional
                all_embeddings.append({
                    "image_id": file_name,
                    "class_name": idx_to_class[labels[i].item()],
                    # Salva o array como lista para facilitar a conversão em string array no SQL
                    "embedding": features[i].tolist(), 
                    "file_path": file_path
                })
                
    # ==========================================
    # 5. Salvamento para Ingestão no BD
    # ==========================================
    print("\nSalvando embeddings para arquivo estruturado...")
    df = pd.DataFrame(all_embeddings)
    
    # Converte a lista do embedding para string no formato "[v1, v2, ...]" esperado pelo pgvector
    df['embedding'] = df['embedding'].apply(lambda x: str(x))
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Concluído! Dados salvos em {OUTPUT_FILE}")
    print(f"Total de imagens processadas: {len(df)}")
    print(f"Dimensão do espaço latente: {len(eval(df['embedding'].iloc[0]))}")

if __name__ == "__main__":
    extract_embeddings()