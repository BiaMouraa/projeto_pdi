# Documentacao do semantic_search_eval.py

Descreve [`semantic_search_eval.py`](../semantic_search_eval.py), o script da Issue 4
que faz a busca semantica de uma imagem de teste e compara as metricas de distancia.

## 1. Objetivo

Receber uma imagem, gerar seu embedding em tempo real (MobileNetV2), consultar o
banco `pgvector` e comparar **Distancia Euclidiana** (`<->`) e **Distancia de
Cosseno** (`<=>`) em termos de vizinhos Top-K, tempo de resposta e coerencia visual.

## 2. Estrutura do arquivo

1. Configuracoes e transformacoes de imagem.
2. Argumentos (`parse_args`).
3. Modelo de features (`load_feature_extractor`).
4. Resolucao da imagem (local, `s3://` ou `--image-id`).
5. Geracao do embedding (`build_embedding`, `to_vector_literal`).
6. Consulta e benchmark (`run_query`, `evaluate_metric`).
7. Escolha da metrica (`choose_metric`) e relatorio (`build_markdown_report`).

## 3. Funcionamento passo a passo

### 3.1 Entrada da imagem

Aceita tres formas:

- `--image /caminho/local.jpg`: arquivo no disco;
- `--image s3://bucket/chave.jpg`: baixa do S3 (ou usa espelho `local_data/`);
- `--image-id NOME.jpg`: busca o `file_path` no banco e resolve a imagem.

`--list-db-samples` lista registros existentes para ajudar a escolher.

### 3.2 Embedding em tempo real

A imagem passa pela mesma transformacao do `extractor.py` (CenterCrop 224 +
Normalize ImageNet) e pela MobileNetV2 sem a camada final, gerando um vetor de
1280 dimensoes convertido para o literal `[v1,v2,...]` aceito pelo `pgvector`.

### 3.3 Consulta Top-K

`run_query` executa, para cada metrica e cada K:

```sql
SELECT image_id, class_name, file_path, embedding <=> %s::vector AS score
FROM flower_embeddings
ORDER BY embedding <=> %s::vector
LIMIT %s;
```

O tempo de cada consulta e medido com `time.perf_counter()`.

### 3.4 Metricas do benchmark

- **Tempo**: media, minimo e maximo em `--runs` repeticoes.
- **Coerencia visual**: fracao dos vizinhos com a mesma classe da consulta
  (quando `--query-class` e informado ou derivado do `--image-id`).

### 3.5 Escolha da metrica final

`choose_metric` prioriza maior coerencia; em empate, escolhe a menor latencia.
O resultado e salvo em `docs/issue4_metric_evaluation.md`.

## 4. Parametros principais

- `--image` / `--image-id`: imagem de teste.
- `--query-class`: classe real (habilita coerencia).
- `--top-k`: lista de K (padrao `5 10`).
- `--runs`: repeticoes do benchmark (padrao `5`).
- `--db-*`: sobrescreve host/porta/credenciais do banco.

## 5. Observacoes

- Sem credenciais AWS, use `--image` local ou espelho `local_data/`.
- A conexao padrao com o banco vem de `pipeline_config.py` (porta 5433).
