import numpy as np

def gerar_kernel_gabor(tamanho, sigma, lambda_, gamma, psi, theta):
    """
    Gera um kernel de Gabor 2D para uma orientação específica[cite: 21, 23, 24, 25, 26, 27].
    """
    # Converte o ângulo de graus para radianos
    theta_rad = np.deg2rad(theta)
    
    # Cria a grade de coordenadas (x, y)
    metade = tamanho // 2
    y, x = np.mgrid[-metade:metade+1, -metade:metade+1]
    
    # Rotação das coordenadas
    x_rot = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    y_rot = -x * np.sin(theta_rad) + y * np.cos(theta_rad)
    
    # Componente Gaussiana (envoltória)
    gauss = np.exp(-(x_rot**2 + (gamma**2 * y_rot**2)) / (2 * sigma**2))
    
    # Componente Senoidal
    senoide = np.cos(2 * np.pi * x_rot / lambda_ + psi)
    
    kernel = gauss * senoide
    
    # Remove a média para evitar resposta a iluminação contínua (opcional mas recomendado em bordas)
    kernel -= np.mean(kernel)
    
    return kernel

def gerar_banco_gabor(config_gabor):
    """
    Retorna uma lista de kernels de Gabor baseada nas orientações fornecidas[cite: 21, 28].
    """
    tamanho = config_gabor['tamanho_mascara']
    if tamanho % 2 == 0:
        raise ValueError(f"tamanho_mascara deve ser ímpar (ex: 31). Recebido: {tamanho}")

    banco = []
    orientacoes = config_gabor['orientacoes_graus']
    
    for theta in orientacoes:
        kernel = gerar_kernel_gabor(
            tamanho=config_gabor['tamanho_mascara'],
            sigma=config_gabor['sigma'],
            lambda_=config_gabor['lambda_'],
            gamma=config_gabor['gamma'],
            psi=config_gabor['psi'],
            theta=theta
        )
        banco.append((theta, kernel))
        
    return banco