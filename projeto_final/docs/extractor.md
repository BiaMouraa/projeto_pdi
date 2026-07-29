# Documentação do extractor de embeddings

Este documento descreve o funcionamento do arquivo [projeto_final/extractor.py](../extractor.py) e explica, de forma passo a passo, os conceitos de visão computacional e aprendizado profundo utilizados no processo de extração de características de imagens.

## 1. Objetivo do script

O script tem como objetivo carregar um conjunto de imagens já processadas, passar cada imagem por uma rede neural convolucional pré-treinada e salvar os vetores de características (embeddings) em um arquivo CSV.

Esses embeddings representam uma descrição numérica compacta da imagem, útil para:

- comparação entre imagens;
- classificação;
- recuperação por similaridade;
- uso em sistemas de busca e bancos vetoriais.

---

## 2. Estrutura do arquivo

O script é organizado em quatro partes principais:

1. Configurações iniciais
2. Pipeline de transformação das imagens
3. Carregamento do modelo
4. Extração e salvamento dos embeddings

---

## 3. Funcionamento passo a passo

### 3.1 Configuração inicial

No início do arquivo, são definidos alguns parâmetros fundamentais:

- PROCESSED_DIR: pasta onde estão as imagens já processadas e organizadas por classe.
- OUTPUT_FILE: local onde o arquivo CSV com os embeddings será salvo.
- BATCH_SIZE: quantidade de imagens processadas de cada vez para economizar memória.

Também é detectado se a máquina possui GPU. Se houver, o processamento é realizado nela para acelerar a execução.

### 3.2 Transformação das imagens

Antes de alimentar a rede neural, as imagens passam por uma etapa de transformação.

O código usa o objeto `transforms.Compose`, que aplica várias operações em sequência:

- `CenterCrop(224)`: corta a imagem para o centro, deixando-a com 224x224 pixels.
- `ToTensor()`: converte a imagem para um tensor PyTorch, com valores entre 0 e 1.
- `Normalize(...)`: padroniza os valores de pixel com base em médias e desvios padrões do dataset ImageNet.

Essa normalização é importante porque redes pré-treinadas foram treinadas com esse padrão. Ela ajuda a melhorar a qualidade da representação extraída.

### 3.3 Carregamento do modelo

A função `get_feature_extractor()` carrega a MobileNetV2 pré-treinada no ImageNet.

A MobileNetV2 é uma rede convolucional leve, muito usada em aplicações de visão computacional por seu bom equilíbrio entre precisão e eficiência.

Na prática, o script remove a camada final de classificação da rede e substitui por uma camada de identidade. Isso faz com que a rede não produza uma classe diretamente, mas sim um vetor de características.

Esse vetor é o chamado embedding, que representa a imagem em um espaço latente de alta dimensão.

### 3.4 Carregamento do dataset

O script usa `datasets.ImageFolder` para carregar automaticamente as imagens a partir da estrutura de pastas.

A organização esperada é:

- pasta principal com subpastas representando as classes;
- cada subpasta contendo as imagens correspondentes.

Exemplo:

- data/processed/rose/image1.jpg
- data/processed/rose/image2.jpg
- data/processed/tulip/image3.jpg

O PyTorch então associa cada pasta a um índice numérico de classe.

### 3.5 Extração dos embeddings

Durante a execução, o código percorre os lotes de imagens e, para cada lote:

1. move as imagens para o dispositivo disponível (GPU ou CPU);
2. passa as imagens pela rede neural;
3. obtém o vetor de características produzido pela rede;
4. converte esses valores para NumPy;
5. associa cada embedding ao nome do arquivo e à classe correspondente.

Esse processo é feito dentro de um bloco `torch.no_grad()`, o que evita o cálculo de gradientes e reduz bastante o uso de memória e o custo computacional.

### 3.6 Salvamento dos dados

No fim, todos os embeddings são organizados em um DataFrame do pandas e salvos em um arquivo CSV.

Cada linha do CSV contém:

- image_id: nome do arquivo de imagem;
- class_name: nome da classe da imagem;
- embedding: vetor numérico da imagem;
- file_path: caminho completo da imagem.

Esse formato facilita a ingestão posterior em bancos de dados vetoriais e sistemas de busca por similaridade.

---

## 4. Conceitos de visão computacional utilizados

### 4.1 Imagem digital

Uma imagem digital é uma matriz de pixels. Cada pixel possui valores numéricos que representam intensidade luminosa ou cores.

Em imagens RGB, por exemplo, cada pixel é composto por três canais:

- vermelho;
- verde;
- azul.

Esses valores são a base para todo o processamento feito pela rede neural.

### 4.2 Redes neurais convolucionais (CNNs)

As Redes Neurais Convolucionais são modelos especializados em processar dados com estrutura espacial, como imagens.

Elas usam filtros convolucionais para detectar padrões locais, como:

- bordas;
- linhas;
- texturas;
- formas;
- objetos parciais.

Esses filtros percorrem a imagem e geram mapas de ativação que destacam determinados padrões.

### 4.3 Convolução

A convolução é a operação principal das CNNs. Ela consiste em aplicar um filtro pequeno sobre a imagem e calcular a resposta local à presença de um padrão.

Essa operação permite que a rede aprenda características hierárquicas:

- camadas iniciais detectam detalhes simples;
- camadas intermediárias detectam padrões mais complexos;
- camadas profundas capturam estruturas semânticas.

### 4.4 Transfer learning

O script usa transfer learning, que consiste em aproveitar uma rede já treinada em um grande conjunto de dados, como o ImageNet, para um novo problema.

Em vez de treinar uma rede do zero, o modelo pré-treinado já possui filtros e representações úteis. Isso reduz tempo, custo computacional e necessidade de muitos dados.

### 4.5 Embeddings

Um embedding é uma representação numérica de alta informação de um objeto, neste caso uma imagem.

Em vez de usar a imagem original em formato de pixels, o modelo transforma a imagem em um vetor de números que captura características relevantes.

Esses vetores podem ser usados para:

- medir similaridade entre imagens;
- realizar agrupamentos;
- alimentar modelos de classificação ou recuperação de informações.

### 4.6 Espaço latente

O espaço latente é o espaço matemático onde os embeddings ficam organizados. Imagens similares tendem a ficar próximas nesse espaço, enquanto imagens diferentes tendem a se afastar.

Essa ideia é central em sistemas de busca por similaridade e representação de dados.

### 4.7 Normalização

A normalização é uma etapa essencial para garantir que os valores de entrada sejam compatíveis com o modelo pré-treinado.

Ela ajuda a estabilizar o treinamento e a inferência, reduzindo variações de contraste, brilho e escala.

---

## 5. Por que esse processo é importante?

A extração de embeddings permite transformar imagens brutas em representações matemáticas úteis. Isso é essencial para aplicações modernas de visão computacional, como:

- reconhecimento de objetos;
- classificação de imagens;
- detecção de similaridade;
- análise de grandes coleções visuais;
- sistemas de recomendação e busca visual.

---

## 6. Resumo

Em resumo, o script:

1. lê imagens de uma pasta organizada por classe;
2. aplica transformações para preparar os dados;
3. usa uma MobileNetV2 pré-treinada;
4. remove a camada final de classificação;
5. extrai um vetor de características para cada imagem;
6. salva esses vetores em um CSV para uso posterior.

Esse processo conecta visão computacional, redes neurais convolucionais e representação de dados em um fluxo simples e eficiente.
