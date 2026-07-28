import json
import os

import numpy as np


def correlacao_2d(imagem, kernel):
    """
    Aplica a correlação espacial 2D em um único canal da imagem.
    """
    img_h, img_w = imagem.shape
    k_h, k_w = kernel.shape

    pad_h, pad_w = k_h // 2, k_w // 2
    img_pad = np.pad(imagem, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    resultado = np.zeros_like(imagem, dtype=np.float32)

    for y in range(k_h):
        for x in range(k_w):
            resultado += img_pad[y:y + img_h, x:x + img_w] * kernel[y, x]

    return resultado


def filtrar_imagem_rgb(imagem_rgb, kernel):
    """Aplica o kernel separadamente nos canais R, G e B."""
    h, w, c = imagem_rgb.shape
    resultado = np.zeros((h, w, c), dtype=np.float32)

    for canal in range(c):
        resultado[:, :, canal] = correlacao_2d(imagem_rgb[:, :, canal], kernel)

    return resultado


def carregar_kernel(caminho):
    """
    Carrega matriz de filtro estático de arquivo .json ou .txt.

    JSON: {"kernel": [[...], ...]} ou matriz direta [[...], ...]
    TXT:  uma linha por linha da matriz, valores separados por espaço
          (linhas vazias ou iniciadas com # são ignoradas)
    """
    caminho = os.path.normpath(caminho)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de filtro não encontrado: '{caminho}'")

    if caminho.endswith('.json'):
        with open(caminho, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kernel = data['kernel'] if isinstance(data, dict) else data
    elif caminho.endswith('.txt'):
        linhas = []
        with open(caminho, 'r', encoding='utf-8') as f:
            for linha in f:
                linha = linha.strip()
                if linha and not linha.startswith('#'):
                    linhas.append([float(x) for x in linha.split()])
        kernel = linhas
    else:
        raise ValueError(f"Formato não suportado (use .json ou .txt): '{caminho}'")

    kernel = np.array(kernel, dtype=np.float32)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Kernel deve ser uma matriz quadrada. Recebido: {kernel.shape}")
    return kernel
