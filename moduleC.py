import numpy as np
from moduleA import filtrar_imagem_rgb

def normalizar_imagem(img):
    """Expansão de histograma para visualização no intervalo [0, 255][cite: 41]."""
    min_val, max_val = img.min(), img.max()
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img - min_val) / (max_val - min_val) * 255
    return img_norm.astype(np.uint8)

def canny_gabor_di_zenzo(img_rgb, banco_gabor):
    h, w, _ = img_rgb.shape
    
    mag_maxima = np.zeros((h, w), dtype=np.float32)
    orientacao_final = np.zeros((h, w), dtype=np.float32)
    
    # 1. Filtragem e 2. Fusão de Canais [cite: 31, 32]
    for theta, kernel in banco_gabor:
        # Filtra os 3 canais simultaneamente
        img_filtrada = filtrar_imagem_rgb(img_rgb, kernel)
        
        # Norma L2 (Di Zenzo): Mag = sqrt(R^2 + G^2 + B^2) [cite: 32, 33]
        mag_theta = np.sqrt(img_filtrada[:,:,0]**2 + img_filtrada[:,:,1]**2 + img_filtrada[:,:,2]**2)
        
        # 3. Redução por Máximo [cite: 34]
        mascara_maior = mag_theta > mag_maxima
        mag_maxima[mascara_maior] = mag_theta[mascara_maior]
        orientacao_final[mascara_maior] = theta  # Guarda o ângulo que gerou o máximo [cite: 35]

    return mag_maxima, orientacao_final

def supressao_nao_maximos(magnitude, orientacao):
    """
    Afina as bordas para 1 pixel de espessura comparando vizinhos na direção do gradiente[cite: 38].
    """
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)
    
    # Converte orientação para ficar entre 0 e 180 graus
    angle = orientacao % 180
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            ang = angle[i, j]
            q, r = 255, 255 # Vizinhos
            
            # Mapeamento do ângulo para os vizinhos ortogonais
            # 0 graus (borda horizontal, gradiente vertical)
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # 45 graus
            elif (22.5 <= ang < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # 90 graus (borda vertical, gradiente horizontal)
            elif (67.5 <= ang < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # 135 graus
            elif (112.5 <= ang < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0
                
    return nms

def histerese(nms, t_low, t_high):
    """
    Aplica conectividade para fechar bordas[cite: 39].
    """
    h, w = nms.shape
    resultado = np.zeros((h, w), dtype=np.uint8)
    
    # Bordas fortes e fracas
    fortes_i, fortes_j = np.where(nms >= t_high)
    fracas_i, fracas_j = np.where((nms >= t_low) & (nms < t_high))
    
    resultado[fortes_i, fortes_j] = 255
    resultado[fracas_i, fracas_j] = 50 # Marcador temporário
    
    # Conecta bordas fracas às fortes
    for i in range(1, h-1):
        for j in range(1, w-1):
            if resultado[i, j] == 50:
                # Verifica se há vizinho forte (8-conectividade)
                if 255 in [resultado[i+1, j-1], resultado[i+1, j], resultado[i+1, j+1],
                           resultado[i, j-1],                      resultado[i, j+1],
                           resultado[i-1, j-1], resultado[i-1, j], resultado[i-1, j+1]]:
                    resultado[i, j] = 255
                else:
                    resultado[i, j] = 0
                    
    # Limpa as que não foram conectadas
    resultado[resultado == 50] = 0
    return resultado