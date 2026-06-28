import numpy as np

def correlacao_2d(imagem, kernel):
    """
    Aplica a correlação espacial 2D em um único canal da imagem[cite: 18].
    Itera sobre o tamanho do kernel para aproveitar a vetorização do NumPy.
    """
    img_h, img_w = imagem.shape
    k_h, k_w = kernel.shape
    
    # Calcula o preenchimento (padding) necessário para manter o tamanho original
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Adiciona zero-padding nas bordas
    img_pad = np.pad(imagem, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    resultado = np.zeros_like(imagem, dtype=np.float32)
    
    # Correlação vetorizada iterando apenas pelo tamanho do kernel (muito mais rápido)
    for y in range(k_h):
        for x in range(k_w):
            resultado += img_pad[y:y+img_h, x:x+img_w] * kernel[y, x]
            
    return resultado

def filtrar_imagem_rgb(imagem_rgb, kernel):
    """
    Aplica o kernel separadamente nos canais R, G e B[cite: 31].
    """
    h, w, c = imagem_rgb.shape
    resultado = np.zeros((h, w, c), dtype=np.float32)
    
    for canal in range(c):
        resultado[:, :, canal] = correlacao_2d(imagem_rgb[:, :, canal], kernel)
        
    return resultado