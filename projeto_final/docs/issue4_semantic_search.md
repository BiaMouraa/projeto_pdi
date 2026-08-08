# Issue 4 - Implementacao e Avaliacao das Metricas de Busca Semantica

Este documento descreve a entrega da Issue 4:

- consultas SQL com operadores do `pgvector`;
- script de busca semantica para gerar embedding de imagem de teste em tempo real;
- retorno Top-5 e Top-10;
- benchmark comparativo de latencia e coerencia visual;
- criterio para selecionar a metrica definitiva.

## 1) Consultas SQL

As consultas foram centralizadas em:

- `sql/semantic_search_queries.sql`

Operadores implementados:

- Distancia Euclidiana: `<->`
- Distancia de Cosseno: `<=>`

As duas consultas aceitam:

- vetor de consulta (`vector`);
- valor de `k` (`int`) para Top-K.

## 2) Script Python de avaliacao

Arquivo:

- `semantic_search_eval.py`

O script executa as etapas abaixo:

1. recebe uma imagem de teste;
2. gera embedding com MobileNetV2 na hora;
3. consulta o PostgreSQL com as duas metricas;
4. retorna resultados para Top-5 e Top-10;
5. mede tempo de resposta (media, minimo, maximo);
6. calcula coerencia visual por classe (quando a classe da imagem de teste e informada);
7. escolhe e registra a metrica recomendada.

## 3) Como executar

No diretorio `projeto_final` (imagens processadas ficam no S3; `data/processed/` local normalmente nao existe).

**Sem credenciais AWS no Mac:** o S3 nao baixa sozinho. Use um `.jpg` local ou copie a imagem para o espelho `local_data/` (mesma estrutura da chave S3, ex.: `local_data/processed/rose/foto.jpg`).

```bash
# Ver ids reais no banco
python3 semantic_search_eval.py --list-db-samples

# Com espelho local (recomendado sem AWS)
python3 semantic_search_eval.py --image-id NOME_DO_ARQUIVO.jpg --top-k 5 10 --runs 5

# Qualquer foto no disco
python3 semantic_search_eval.py --image /caminho/para/flor.jpg --query-class rose --top-k 5 10 --runs 5
```

Com AWS configurada, URI S3 tambem funciona:

```bash
python3 semantic_search_eval.py --image s3://iris-cv-latente-data/processed/rose/arquivo.jpg --query-class rose --top-k 5 10 --runs 5
```

Parametros principais:

- `--image` ou `--image-id`: caminho local, URI `s3://...`, ou `image_id` na tabela `flower_embeddings` (obrigatorio um dos dois);
- `--query-class`: classe real da imagem (opcional, habilita coerencia por classe);
- `--top-k`: lista de valores de K (padrao: `5 10`);
- `--runs`: repeticoes por consulta para benchmark (padrao: `5`).

## 4) Criterio de selecao da metrica final

Regra automatica implementada no script:

1. prioriza maior coerencia visual media (Top-K);
2. em empate de coerencia, escolhe menor latencia media;
3. se coerencia nao estiver disponivel (sem `--query-class`), escolhe menor latencia media.

## 5) Relatorio final da execucao

A cada rodada, o script gera:

- `docs/issue4_metric_evaluation.md`

Este arquivo inclui:

- SQL usada para cada metrica;
- tempos por Top-K;
- lista de vizinhos retornados;
- metrica definitiva escolhida com justificativa.

## 6) Metrica definitiva para a versao final

Para a versao final do projeto, a metrica padrao definida e **Distancia de Cosseno (`<=>`)**.

Motivo pratico:

- embeddings de CNN costumam variar em norma (magnitude), e o cosseno reduz esse efeito;
- em busca semantica de imagens, a orientacao no espaco latente tende a representar melhor a semelhanca visual;
- o benchmark implementado continua disponivel para revalidacao periodica com novas amostras.
