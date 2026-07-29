import os
import cv2
import kagglehub
from pathlib import Path

# ==========================================
# 1. Configurações Iniciais
# ==========================================
DATASET_URI = "cantonioupao/oxford-flower-17categories-labelled"
PROCESSED_DIR = "data/processed"

TARGET_SIZE = (256, 256)

def download_dataset():
    print("Iniciando o download/leitura do dataset Oxford-17 via Kagglehub...")
    path = kagglehub.dataset_download(DATASET_URI)
    print(f"Download/Leitura concluída! Arquivos brutos localizados em:\n{path}\n")
    return path

def preprocess_images(input_dir, output_dir, target_size):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True)

    print(f"Iniciando pré-processamento. Buscando imagens recursivamente em: {input_path.name}...")
    
    # Dicionário para guardar a contagem de imagens por classe
    count_dict = {}

    # O rglob("*.*") entra em TODAS as subpastas automaticamente
    for img_file in input_path.rglob("*.*"):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        # O nome da classe será o nome da pasta imediatamente acima da imagem
        class_name = img_file.parent.name
        
        # Ignora se a imagem estiver solta na pasta raiz sem classe
        if class_name == input_path.name:
            continue
            
        class_output_dir = output_path / class_name
        class_output_dir.mkdir(exist_ok=True)
        
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            img_bgr_to_save = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            
            save_path = class_output_dir / img_file.name
            cv2.imwrite(str(save_path), img_bgr_to_save)
            
            # Atualiza o contador da classe
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
            
        except Exception as e:
            print(f"Erro ao processar {img_file.name}: {e}")

    # Exibe o resumo final das classes processadas
    print("\nResumo do processamento:")
    for c_name, c_count in sorted(count_dict.items()):
        print(f" -> {c_count} imagens processadas na classe '{c_name}'.")

    print(f"\nPré-processamento finalizado! Dataset salvo em: {output_path.absolute()}")

if __name__ == "__main__":
    raw_dataset_path = download_dataset()
    preprocess_images(raw_dataset_path, PROCESSED_DIR, TARGET_SIZE)