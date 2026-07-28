from collections import deque

import numpy as np

from projeto1.moduleA import carregar_kernel, correlacao_2d, filtrar_imagem_rgb


def normalizar_imagem(img):
    """Expansão de histograma para visualização no intervalo [0, 255]."""
    min_val, max_val = img.min(), img.max()
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img - min_val) / (max_val - min_val) * 255
    return img_norm.astype(np.uint8)


def rgb_para_cinza_manual(img_rgb):
    """Conversão manual para tons de cinza: Y = 0.299R + 0.587G + 0.114B."""
    return (
        0.299 * img_rgb[:, :, 0]
        + 0.587 * img_rgb[:, :, 1]
        + 0.114 * img_rgb[:, :, 2]
    )


def canny_gabor_di_zenzo(img_rgb, banco_gabor):
    """Canny modificado: Gabor por orientação + fusão L2 (Di Zenzo) + máximo."""
    h, w, _ = img_rgb.shape

    mag_maxima = np.zeros((h, w), dtype=np.float32)
    orientacao_final = np.zeros((h, w), dtype=np.float32)

    for theta, kernel in banco_gabor:
        img_filtrada = filtrar_imagem_rgb(img_rgb, kernel)
        mag_theta = np.sqrt(
            img_filtrada[:, :, 0] ** 2
            + img_filtrada[:, :, 1] ** 2
            + img_filtrada[:, :, 2] ** 2
        )

        mascara_maior = mag_theta > mag_maxima
        mag_maxima[mascara_maior] = mag_theta[mascara_maior]
        orientacao_final[mascara_maior] = theta

    return mag_maxima, orientacao_final


def canny_gabor_escalar(img_rgb, banco_gabor):
    """
    Gabor tradicional (escalar): aplica banco de Gabor apenas no canal cinza manual.
    Usado no Experimento 1 para comparar sensibilidade paramétrica com o modificado.
    """
    cinza = rgb_para_cinza_manual(img_rgb)
    h, w = cinza.shape

    mag_maxima = np.zeros((h, w), dtype=np.float32)
    orientacao_final = np.zeros((h, w), dtype=np.float32)

    for theta, kernel in banco_gabor:
        resposta = correlacao_2d(cinza, kernel)
        mag_theta = np.abs(resposta)

        mascara_maior = mag_theta > mag_maxima
        mag_maxima[mascara_maior] = mag_theta[mascara_maior]
        orientacao_final[mascara_maior] = theta

    return mag_maxima, orientacao_final


def supressao_nao_maximos(magnitude, orientacao):
    """Afina bordas para ~1 pixel comparando vizinhos na direção do gradiente."""
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)
    angle = orientacao % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            ang = angle[i, j]
            q, r = 255.0, 255.0

            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= ang < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= ang < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= ang < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]

    return nms


def histerese(nms, t_low, t_high):
    """
    Limiarização por histerese com propagação iterativa (BFS).
    Bordas fracas conectadas a fortes (direta ou em cadeia) são mantidas.
    """
    h, w = nms.shape
    STRONG = 255
    WEAK = 128

    resultado = np.zeros((h, w), dtype=np.uint8)
    resultado[nms >= t_high] = STRONG
    resultado[(nms >= t_low) & (nms < t_high)] = WEAK

    fila = deque(zip(*np.where(resultado == STRONG)))
    while fila:
        i, j = fila.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and resultado[ni, nj] == WEAK:
                    resultado[ni, nj] = STRONG
                    fila.append((ni, nj))

    resultado[resultado != STRONG] = 0
    return resultado


def canny_tradicional(img_rgb, cfg_tradicional):
    """
    Pipeline Canny clássico: cinza manual → Gauss (arquivo) → Sobel (arquivo) → NMS → histerese.
    """
    cinza = rgb_para_cinza_manual(img_rgb)

    kernel_gauss = carregar_kernel(cfg_tradicional['filtro_gaussiana'])
    kernel_sobel_x = carregar_kernel(cfg_tradicional['filtro_sobel_x'])
    kernel_sobel_y = carregar_kernel(cfg_tradicional['filtro_sobel_y'])

    cinza_suav = correlacao_2d(cinza, kernel_gauss)
    gx = correlacao_2d(cinza_suav, kernel_sobel_x)
    gy = correlacao_2d(cinza_suav, kernel_sobel_y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientacao = np.rad2deg(np.arctan2(gy, gx)) % 180

    nms = supressao_nao_maximos(magnitude, orientacao)
    borda_final = histerese(nms, cfg_tradicional['t_low'], cfg_tradicional['t_high'])

    return magnitude, orientacao, nms, borda_final
