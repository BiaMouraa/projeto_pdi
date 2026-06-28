# Pipeline de Canny Modificado com Gabor-Di Zenzo

Este projeto implementa um pipeline de detecção de bordas que combina filtros de Gabor com o método de fusão vetorial de Di Zenzo, seguido por supressão de não-máximos e limiarização por histerese. O objetivo é detectar bordas em imagens coloridas de forma robusta e bem afinada.

## Estrutura do projeto

- `main.py`: script principal. Recebe uma imagem de entrada, carrega a configuração de `config.json`, cria o banco de filtros de Gabor, executa o pipeline de detecção e grava os resultados em `results/<nome_da_imagem>/`.
- `moduleA.py`: contém funções de correlação 2D e filtragem separada para cada canal RGB.
- `moduleB.py`: gera kernels de Gabor e constrói um banco de filtros para múltiplas orientações.
- `moduleC.py`: implementa o pipeline de Canny modificado com fusão L2 no espaço RGB, supressão de não-máximos e histerese.
- `config.json`: parâmetros do filtro de Gabor e valores de limiarização para histerese.
- `images/`: pasta onde podem ser armazenadas imagens de teste.
- `results/`: pasta de saída onde os resultados do processamento são salvos.

## Como usar

1. Posicione a imagem de teste em `images/` ou informe o caminho completo.
2. Execute o script a partir do diretório do projeto:

```bash
python main.py images/GrayAndMagenta.png
```

3. O processamento gera automaticamente a pasta `results/<nome_da_imagem>/` com três imagens:

- `1_magnitude_normalizada.png`: magnitude dos gradientes após filtragem e fusão vetorial.
- `2_nms.png`: resultado da supressão de não-máximos.
- `3_borda_final_histerese.png`: bordas finais após histerese.

## Parâmetros e funções principais

### `config.json`

- `gabor.tamanho_mascara`: tamanho do kernel de Gabor (número ímpar).
- `gabor.sigma`: desvio padrão da gaussiana do kernel.
- `gabor.lambda_`: comprimento de onda da componente senoidal.
- `gabor.gamma`: razão de aspecto da gaussiana.
- `gabor.psi`: fase do componente senoidal.
- `gabor.orientacoes_graus`: lista de ângulos em graus para gerar o banco de kernels de Gabor.
- `histerese.t_low`: limiar inferior para bordas fracas.
- `histerese.t_high`: limiar superior para bordas fortes.

### `moduleA.py`

- `correlacao_2d(imagem, kernel)`: aplica correlação 2D em uma imagem de um canal, com zero-padding e vetorização.
- `filtrar_imagem_rgb(imagem_rgb, kernel)`: filtra cada canal RGB separadamente usando `correlacao_2d`.

### `moduleB.py`

- `gerar_kernel_gabor(tamanho, sigma, lambda_, gamma, psi, theta)`: cria um kernel 2D de Gabor para um ângulo `theta`.
- `gerar_banco_gabor(config_gabor)`: constrói uma lista de kernels de Gabor para todas as orientações especificadas.

### `moduleC.py`

- `normalizar_imagem(img)`: normaliza a imagem para o intervalo `[0, 255]` para salvar resultados em formato PNG.
- `canny_gabor_di_zenzo(img_rgb, banco_gabor)`: aplica cada filtro de Gabor no RGB, calcula magnitude L2 e seleciona a maior magnitude e a orientação correspondente.
- `supressao_nao_maximos(magnitude, orientacao)`: afina as bordas comparando pixels vizinhos ao longo da direção do gradiente.
- `histerese(nms, t_low, t_high)`: classifica pixels como fortes ou fracos e conserva apenas bordas fracas conectadas a bordas fortes.

## Resultado esperado

O pipeline produz uma versão afinada das bordas da imagem de entrada em alta qualidade, aproveitando informações de cor e orientação para reduzir ruídos e preservar contornos significativos.

## Observações

- A imagem de entrada deve estar em um formato suportado pelo OpenCV (por exemplo, PNG ou JPG).
- Os resultados são gravados em `results/<nome_da_imagem>/` e a pasta é criada automaticamente se não existir.
- Ajustar `config.json` permite controlar a sensibilidade do filtro de Gabor e a rigidez da limiarização por histerese.
