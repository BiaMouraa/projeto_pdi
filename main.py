import json
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from moduleB import gerar_banco_gabor
from moduleC import canny_gabor_di_zenzo, supressao_nao_maximos, histerese, normalizar_imagem

def main():
    # 1. Configuração do parser de argumentos para o terminal
    parser = argparse.ArgumentParser(
        description="Pipeline de Canny Modificado (Gabor-Di Zenzo) com processamento vetorial."
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Caminho para a imagem de teste (ex: GrayAndMagenta.png ou dados/imagem.png)"
    )
    args = parser.parse_args()

    # 2. Carrega a configuração JSON
    if not os.path.exists('config.json'):
        print("Erro: O arquivo 'config.json' não foi encontrado no diretório atual.")
        return

    with open('config.json', 'r') as f:
        config = json.load(f)

    # 3. Carrega a imagem passada como parâmetro
    image_path = args.image_path
    img_bgr = cv2.imread(image_path) 
    if img_bgr is None:
        print(f"Erro: Não foi possível abrir ou encontrar a imagem no caminho: '{image_path}'")
        return
        
    # Conversão de BGR (padrão do OpenCV) para RGB em float32 para os cálculos
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 4. Gera o Banco de Filtros de Gabor
    banco_gabor = gerar_banco_gabor(config['gabor'])

    # 5. Execução do Pipeline do Canny Modificado
    print(f"A processar a imagem: {image_path}")
    print("-> A executar filtragem espacial e fusão vetorial L2...")
    magnitude, orientacao = canny_gabor_di_zenzo(img_rgb, banco_gabor)
    
    print("-> A aplicar Supressão de Não-Máximos (Afinamento NMS)...")
    nms = supressao_nao_maximos(magnitude, orientacao)
    
    print("-> A aplicar Limiarização por Histerese...")
    borda_final = histerese(nms, config['histerese']['t_low'], config['histerese']['t_high'])

    # 6. Criação automática da estrutura de pastas
    # Path(image_path).stem extrai apenas o nome do ficheiro, ignorando o caminho e a extensão.
    # Exemplo: "diretorio/imagens/GrayAndMagenta.png" vira "GrayAndMagenta"
    nome_imagem_original = Path(image_path).stem
    
    # Define o caminho da pasta destino: results/nome_da_imagem_original
    pasta_destino = os.path.join("results", nome_imagem_original)
    
    # Cria a pasta e subpastas caso não existam (o parâmetro exist_ok evita erros se já existir)
    os.makedirs(pasta_destino, exist_ok=True)

    # 7. Definição dos caminhos e gravação das imagens resultantes
    caminho_mag = os.path.join(pasta_destino, "1_magnitude_normalizada.png")
    caminho_nms = os.path.join(pasta_destino, "2_nms.png")
    caminho_borda = os.path.join(pasta_destino, "3_borda_final_histerese.png")

    cv2.imwrite(caminho_mag, normalizar_imagem(magnitude))
    cv2.imwrite(caminho_nms, normalizar_imagem(nms))
    cv2.imwrite(caminho_borda, borda_final)
    
    print("\n[SUCESSO] Processamento concluído!")
    print(f"Os resultados foram guardados com sucesso na pasta:\n -> {pasta_destino}/")

if __name__ == "__main__":
    main()