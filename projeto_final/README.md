# Projeto Final - Pipeline de Visão Computacional

Este projeto realiza um pipeline completo de visão computacional para extração de embeddings de imagens e ingestão em um banco de dados vetorial PostgreSQL com `pgvector`.

## Estrutura principal

- `preprocess.py`: faz o download e pré-processamento das imagens.
- `extractor.py`: extrai embeddings das imagens processadas usando MobileNetV2.
- `load_embeddings.py`: carrega os embeddings no banco PostgreSQL + `pgvector`.
- `docker-compose.yml`: orquestra um container PostgreSQL com `pgvector`.
- `requirements.txt`: dependências Python necessárias.
- `docs/`: documentação de cada etapa do pipeline.

## Passo a passo para execução

### 1. Instalar dependências

No diretório `projeto_final`, crie e ative seu ambiente virtual Python e instale as dependências:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pré-processar as imagens

Execute o script de preprocessamento para baixar o dataset Oxford Flower 17, reorganizar as imagens por classe e redimensioná-las:

```bash
python preprocess.py
```

Isso criará a pasta `data/processed` com as imagens tratadas.

### 3. Extrair embeddings

Com as imagens pré-processadas, execute o extractor para gerar os vetores de características:

```bash
python extractor.py
```

O resultado será salvo em `data/embeddings.csv`.

### 4. Subir o banco de dados com Docker Compose

Inicie o PostgreSQL com `pgvector` usando o Docker Compose:

```bash
docker compose up -d
```

Espere até o serviço `db_vector` estar pronto para aceitar conexões na porta `5432`.

### 5. Carregar embeddings no banco

Execute o script de ingestão para inserir os embeddings no PostgreSQL:

```bash
python load_embeddings.py
```

Isso criará a tabela `flower_embeddings`, inserirá os dados e construirá um índice HNSW para buscas por similaridade.

## Verificação e próximos passos

- Verifique se o arquivo `data/embeddings.csv` foi gerado corretamente.
- Confirme que o container PostgreSQL está rodando e ouvindo em `localhost:5432`.
- Consulte `docs/preprocess.md`, `docs/extractor.md` e `docs/load_embeddings.md` para entender cada etapa em detalhe.

## Observações

- Se usar Linux ou WSL, o comando para ativar o ambiente virtual pode ser `source .venv/bin/activate`.
- Caso altere a configuração do banco, atualize também `load_embeddings.py` e `docker-compose.yml`.

## Referências rápidas

- `python preprocess.py`
- `python extractor.py`
- `docker compose up -d`
- `python load_embeddings.py`
