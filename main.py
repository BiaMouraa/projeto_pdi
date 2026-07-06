import argparse
import json
import os
from pathlib import Path

from pipeline import (
    carregar_imagem_rgb,
    executar_canny_tradicional,
    executar_pipeline_modificado,
    pasta_canny_tradicional,
    resolver_pasta_destino,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Canny Modificado (Gabor-Di Zenzo) + Canny tradicional."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Caminho para a imagem de teste (ex: images/GrayAndMagenta.png)",
    )
    parser.add_argument(
        "--sem-tradicional",
        action="store_true",
        help="Não executa o Canny tradicional (apenas o modificado).",
    )
    args = parser.parse_args()

    if not os.path.exists('config.json'):
        print("Erro: O arquivo 'config.json' não foi encontrado no diretório atual.")
        return

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    image_path = args.image_path
    try:
        img_rgb = carregar_imagem_rgb(image_path)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return

    nome_imagem = Path(image_path).stem
    pasta_modificado = resolver_pasta_destino(nome_imagem, config)

    print(f"A processar: {image_path}")
    executar_pipeline_modificado(img_rgb, config, pasta_modificado)
    print(f"  Modificado  -> {pasta_modificado}/")

    if not args.sem_tradicional:
        pasta_trad = pasta_canny_tradicional(nome_imagem)
        executar_canny_tradicional(img_rgb, config, pasta_trad)
        print(f"  Tradicional -> {pasta_trad}/")

    print("\n[SUCESSO] Processamento concluído!")


if __name__ == "__main__":
    main()
