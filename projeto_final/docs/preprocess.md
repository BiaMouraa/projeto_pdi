# Documentação do preprocess

Este documento descreve o funcionamento do arquivo [projeto_final/preprocess.py](../preprocess.py) e explica os conceitos de preprocessamento de imagens usados neste projeto.

## 1. Objetivo do script

O script faz o download do dataset Oxford Flower 17 usando o Kagglehub, e aplica transformações nas imagens para torná-las consistentes para etapas posteriores de extração de embeddings.

O objetivo principal é:

- obter os arquivos de imagem brutos;
- detectar e organizar classes;
- padronizar tamanho e formato;
- salvar uma versão tratada do dataset em `data/processed`.

---

## 2. Estrutura do arquivo

O arquivo contém as seguintes partes:

1. configurações iniciais;
2. download do dataset;
3. rotina de preprocessamento;
4. pipeline de leitura, conversão e salvamento das imagens.

---

## 3. Funcionamento passo a passo

### 3.1 Configurações iniciais

As constantes definidas no início do arquivo são:

- `DATASET_URI`: identificador do dataset no Kagglehub (`cantonioupao/oxford-flower-17categories-labelled`);
- `PROCESSED_DIR`: pasta de destino para as imagens tratadas;
- `TARGET_SIZE`: tamanho final das imagens, definido como `(256, 256)`.

### 3.2 Download do dataset

A função `download_dataset()` utiliza a biblioteca `kagglehub` para baixar ou ler os arquivos do dataset.

Ela retorna o caminho local onde os arquivos brutos foram armazenados.

### 3.3 Pré-processamento das imagens

A função `preprocess_images(input_dir, output_dir, target_size)` faz o seguinte:

1. cria a pasta de saída se ela não existir;
2. percorre recursivamente todos os arquivos na pasta de entrada;
3. ignora arquivos que não tenham extensões de imagem suportadas (`.jpg`, `.jpeg`, `.png`);
4. determina o nome da classe a partir da pasta pai da imagem;
5. lê a imagem usando OpenCV;
6. converte de BGR para RGB;
7. redimensiona para `TARGET_SIZE`;
8. converte de volta para BGR para salvar com OpenCV;
9. salva a imagem no diretório de classe correspondente.

### 3.4 Organização por classes

As imagens tratadas são salvas em uma estrutura de pastas organizada por classe. Por exemplo:

- `data/processed/daisy/image1.jpg`
- `data/processed/rose/image2.jpg`

Essa estrutura é compatível com a leitura posterior por `torchvision.datasets.ImageFolder`, usada no script de extração de embeddings.

### 3.5 Resumo do processamento

Ao final, o script imprime um resumo indicando quantas imagens foram processadas por classe e o caminho final do dataset pré-processado.

---

## 4. Conceitos e teoria de preprocessamento

### 4.1 Por que preprocessar imagens?

Pré-processar imagens é importante para garantir que todos os dados de entrada tenham características consistentes antes de serem usados em modelos de aprendizado de máquina.

Sem uma etapa de preprocessamento, o modelo pode receber imagens com tamanhos, formatos e canais diferentes, o que prejudica o desempenho.

### 4.2 Redimensionamento

Redimensionar todas as imagens para o mesmo tamanho (`256x256`) é uma prática comum em visão computacional.

Isso garante que a rede neural receba entradas com dimensões fixas e permite processar os dados em lotes.

### 4.3 Conversão de cores BGR para RGB

O OpenCV lê imagens no formato BGR, enquanto muitas bibliotecas e modelos de visão usam o formato RGB.

A conversão `cv2.COLOR_BGR2RGB` garante que as cores estejam no padrão correto para possíveis visualizações ou transformações futuras.

### 4.4 Estrutura de pastas por classe

Organizar imagens em subpastas nomeadas pelas classes permite que bibliotecas como `ImageFolder` inferiram automaticamente os rótulos.

Isso facilita o pipeline de treinamento, validação e extração de recursos.

### 4.5 Robustez e filtragem

O script ignora arquivos que não são imagens e trata casos em que a leitura falha (`img is None`).

Essa verificação previne erros durante o processamento e permite que o fluxo continue mesmo com arquivos inválidos.

---

## 5. Por que essa etapa é importante?

A etapa de preprocessamento é o primeiro passo para criar um dataset limpo e uniforme.

Ela prepara as imagens para serem consumidas por redes neurais e garante que o pipeline seguinte (`extractor.py`) trabalhe com dados confiáveis.

---

## 6. Resumo final

`preprocess.py` realiza:

1. download ou leitura do dataset do Kagglehub;
2. varredura recursiva das imagens brutas;
3. filtragem de formatos compatíveis;
4. conversão de cores e redimensionamento;
5. salvamento em uma estrutura organizada por classe.

Esse processo transforma imagens brutas em um dataset padronizado, pronto para extração de embeddings e análises posteriores.
