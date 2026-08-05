# Documentação do load_embeddings

Este documento descreve o funcionamento do arquivo [projeto_final/load_embeddings.py](../load_embeddings.py) e explica os conceitos de ingestão de embeddings em banco de dados vetoriais usados pelo script.

## 1. Objetivo do script

O objetivo é carregar embeddings previamente extraídos de imagens e inseri-los em um banco de dados PostgreSQL com a extensão `pgvector`.

Essa ingestão permite consultar imagens por similaridade no espaço vetorial, usando índices especializados para buscas rápidas.

---

## 2. Estrutura do arquivo

O arquivo está organizado em partes principais:

1. configurações de conexão com o banco;
2. definições das queries SQL;
3. leitura do CSV de embeddings;
4. inserção dos dados no banco;
5. criação dos índices vetoriais.

---

## 3. Funcionamento passo a passo

### 3.1 Configurações de conexão

A constante `DB_CONFIG` contém os parâmetros necessários para conectar ao banco de dados PostgreSQL:

- `dbname`: nome do banco de dados;
- `user`: usuário do PostgreSQL;
- `password`: senha do usuário;
- `host`: endereço do servidor;
- `port`: porta de conexão.

No contexto do projeto, o banco é esperado em execução via Docker no `localhost:5432`.

### 3.2 Arquivo de entrada

O script lê o arquivo CSV definido em `CSV_FILE`.

Esse arquivo é gerado por `extractor.py` e contém, para cada imagem:

- `image_id`: nome do arquivo;
- `class_name`: classe da imagem;
- `file_path`: caminho da imagem;
- `embedding`: vetor de características.

### 3.3 Setup do banco de dados

O script executa a query `SQL_SETUP`, que faz o seguinte:

- habilita a extensão `vector` do PostgreSQL (`CREATE EXTENSION IF NOT EXISTS vector;`);
- remove a tabela anterior `flower_embeddings` quando existe (`DROP TABLE IF EXISTS`), facilitando testes repetidos;
- cria a tabela `flower_embeddings` com os campos necessários.

A tabela usa o tipo `VECTOR(1280)` para o campo `embedding`, compatível com vetores da MobileNetV2.

### 3.4 Inserção dos dados

A função `carregar_dados_no_banco()` abre a conexão com o banco e lê o CSV usando o pandas.

Em seguida, percorre cada linha com `tqdm` para exibir uma barra de progresso e insere os valores na tabela com `cursor.execute(SQL_INSERT, (...))`.

Dessa forma, cada embedding fica armazenado como um vetor nativo do PostgreSQL.

### 3.5 Criação de índices vetoriais

Após a ingestão, o script cria dois índices HNSW:

```sql
CREATE INDEX IF NOT EXISTS flower_embeddings_embedding_cosine_idx
ON flower_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS flower_embeddings_embedding_l2_idx
ON flower_embeddings USING hnsw (embedding vector_l2_ops);
```

Esses índices aceleram buscas por similaridade tanto em distância de cosseno quanto em distância euclidiana.

---

## 4. Conceitos e teoria

### 4.1 Banco de dados vetorial

Um banco de dados vetorial é otimizado para armazenar vetores numéricos e realizar consultas como "quais vetores são mais parecidos com este vetor de consulta?".

No projeto, o PostgreSQL com a extensão `pgvector` serve como repositório desses embeddings.

### 4.2 Tipos vetoriais

O tipo `VECTOR(1280)` armazena vetores de dimensão fixa.

A dimensão 1280 vem do embedding retornado pela MobileNetV2 quando removemos a camada de classificação final.

### 4.3 Ingestão de dados

Ingestão é o processo de ler dados de uma fonte (CSV, arquivo, API) e salvá-los em um sistema persistente.

Aqui, o CSV é a fonte e o PostgreSQL é o destino.

### 4.4 Similaridade vetorial

Para comparar imagens, calculamos a similaridade entre seus embeddings.

A métrica usada pelo índice é a similaridade de cosseno, que mede o ângulo entre vetores.

- valores próximos de 1 significam vetores muito semelhantes;
- valores próximos de 0 significam vetores menos semelhantes.

### 4.5 Índice HNSW

HNSW (Hierarchical Navigable Small World) é uma estrutura de índice eficiente para busca aproximada de vizinhos mais próximos (ANN).

Ele permite consultas rápidas em espaços de alta dimensão, como embeddings de imagens, com boa precisão e baixa latência.

### 4.6 Extensão `pgvector`

A extensão `pgvector` adiciona suporte para vetores ao PostgreSQL e fornece operadores e índices voltados a similaridade.

No script, ela é usada tanto no tipo de coluna (`VECTOR`) quanto nos índices (`vector_cosine_ops` e `vector_l2_ops`).

---

## 5. Por que usar esse fluxo?

O fluxo de `load_embeddings.py` é útil porque:

- permite transformar embeddings em dados pesquisáveis;
- prepara o sistema para consultas de similaridade visual;
- cria uma base estruturada com metadados de imagens;
- usa recursos de banco de dados para escalar consultas.

---

## 6. Resumo

O script realiza o seguinte:

1. conecta ao PostgreSQL;
2. prepara a tabela e habilita a extensão `pgvector`;
3. lê o CSV de embeddings;
4. insere cada registro na tabela;
5. cria índices HNSW para buscas rápidas com cosseno e euclidiana.

Com isso, os embeddings passam a estar acessíveis para aplicações que buscam imagens semelhantes ou realizam análise de similaridade no banco de dados.
