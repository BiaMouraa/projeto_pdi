import json
import os

import cv2

from projeto1.moduleB import gerar_banco_gabor
from projeto1.moduleC import (
    canny_gabor_di_zenzo,
    canny_gabor_escalar,
    canny_tradicional,
    histerese,
    normalizar_imagem,
    supressao_nao_maximos,
)


def gerar_nome_pasta_config(config):
    g = config['gabor']
    h = config['histerese']
    return (
        f"mask{g['tamanho_mascara']}_"
        f"sigma{g['sigma']}_"
        f"lambda{g['lambda_']}_"
        f"gamma{g['gamma']}_"
        f"psi{g['psi']}_"
        f"norient{len(g['orientacoes_graus'])}_"
        f"tlow{h['t_low']}_"
        f"thigh{h['t_high']}"
    )


def resolver_pasta_destino(nome_imagem, config, sufixo=None):
    nome = gerar_nome_pasta_config(config)
    if sufixo:
        nome = f"{nome}_{sufixo}"
    pasta_base = os.path.join("results", nome_imagem, nome)
    if not os.path.exists(pasta_base):
        return pasta_base

    n = 2
    while os.path.exists(f"{pasta_base}_run{n}"):
        n += 1
    return f"{pasta_base}_run{n}"


def pasta_canny_tradicional(nome_imagem):
    return os.path.join("results", nome_imagem, "canny_tradicional")


def salvar_config(pasta_destino, config):
    with open(os.path.join(pasta_destino, 'config_usado.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


def executar_pipeline_modificado(img_rgb, config, pasta_destino, verbose=True):
    """Executa Canny modificado (Di Zenzo) + Gabor escalar. Salva arquivos 1–3 e 7–9."""
    if verbose:
        print("  -> Gabor modificado (Di Zenzo)...")
    banco_gabor = gerar_banco_gabor(config['gabor'])
    magnitude, orientacao = canny_gabor_di_zenzo(img_rgb, banco_gabor)

    if verbose:
        print("  -> Gabor escalar (cinza)...")
    mag_escalar, orient_escalar = canny_gabor_escalar(img_rgb, banco_gabor)

    if verbose:
        print("  -> NMS + histerese (modificado)...")
    nms = supressao_nao_maximos(magnitude, orientacao)
    borda_final = histerese(
        nms,
        config['histerese']['t_low'],
        config['histerese']['t_high'],
    )

    if verbose:
        print("  -> NMS + histerese (Gabor escalar)...")
    nms_escalar = supressao_nao_maximos(mag_escalar, orient_escalar)
    borda_escalar = histerese(
        nms_escalar,
        config['histerese']['t_low'],
        config['histerese']['t_high'],
    )

    os.makedirs(pasta_destino, exist_ok=True)
    salvar_config(pasta_destino, config)

    cv2.imwrite(os.path.join(pasta_destino, "1_magnitude_di_zenzo.png"), normalizar_imagem(magnitude))
    cv2.imwrite(os.path.join(pasta_destino, "2_nms_di_zenzo.png"), normalizar_imagem(nms))
    cv2.imwrite(os.path.join(pasta_destino, "3_borda_final_di_zenzo.png"), borda_final)
    cv2.imwrite(os.path.join(pasta_destino, "7_gabor_escalar_magnitude.png"), normalizar_imagem(mag_escalar))
    cv2.imwrite(os.path.join(pasta_destino, "8_gabor_escalar_nms.png"), normalizar_imagem(nms_escalar))
    cv2.imwrite(os.path.join(pasta_destino, "9_gabor_escalar_borda_final.png"), borda_escalar)

    return pasta_destino


def executar_canny_tradicional(img_rgb, config, pasta_destino, verbose=True):
    """Executa Canny tradicional (Sobel) e salva arquivos 4–6."""
    if verbose:
        print("  -> Canny tradicional (cinza + Sobel via arquivos)...")
    cfg = config['canny_tradicional']
    magnitude, _, nms, borda_final = canny_tradicional(img_rgb, cfg)

    os.makedirs(pasta_destino, exist_ok=True)
    salvar_config(pasta_destino, config)

    cv2.imwrite(os.path.join(pasta_destino, "4_tradicional_magnitude.png"), normalizar_imagem(magnitude))
    cv2.imwrite(os.path.join(pasta_destino, "5_tradicional_nms.png"), normalizar_imagem(nms))
    cv2.imwrite(os.path.join(pasta_destino, "6_tradicional_borda_final.png"), borda_final)

    return pasta_destino


def carregar_imagem_rgb(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: '{image_path}'")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32')
