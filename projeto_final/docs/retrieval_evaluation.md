# Documentacao do retrieval_evaluation.py

Descreve o funcionamento de [`retrieval_evaluation.py`](../retrieval_evaluation.py),
o script da Issue 5 que avalia o desempenho da recuperacao de imagens.

## 1. Objetivo

Medir Precision/Recall da busca Top-K no `pgvector`, focar em classes criticas,
analisar falsos positivos e gerar graficos, produzindo um relatorio pronto para a
secao de Resultados e Discussao.

## 2. Estrutura do arquivo

1. Argumentos de linha de comando (`parse_args`).
2. Acesso ao banco (`get_class_counts`, `fetch_topk_all`).
3. Metricas (`precision_recall_at_k`, `build_pr_curve`).
4. Matriz de confusao (`build_topk_confusion`).
5. Falsos positivos (`analyze_false_positives`).
6. Graficos (`plot_pr_curve`, `plot_confusion`, `make_montage`).
7. Relatorio (`write_report`).
8. Orquestracao (`main`).

## 3. Funcionamento passo a passo

### 3.1 Recuperacao de vizinhos em lote

`fetch_topk_all` usa uma unica consulta SQL com `CROSS JOIN LATERAL`:

```sql
SELECT q.id, q.image_id, q.class_name, q.file_path,
       n.image_id, n.class_name, n.file_path, n.score
FROM flower_embeddings q
CROSS JOIN LATERAL (
    SELECT image_id, class_name, file_path,
           embedding <=> q.embedding AS score
    FROM flower_embeddings
    WHERE id <> q.id
    ORDER BY embedding <=> q.embedding
    LIMIT %s
) n
ORDER BY q.id, n.score;
```

Assim, para **cada** imagem do banco calculamos seus K vizinhos mais proximos de uma
vez, aproveitando o indice HNSW. O operador (`<->` ou `<=>`) depende de `--metric`.

### 3.2 Precision e Recall

`precision_recall_at_k` percorre cada consulta, conta quantos dos K vizinhos sao da
mesma classe e calcula Precision@K e Recall@K. Depois agrega em:

- media por classe (`class_precision`, `class_recall`);
- macro (media das classes) e micro (media das consultas).

`build_pr_curve` repete isso para K de 1 ate `--max-k`, gerando os pontos da curva.

### 3.3 Matriz de confusao Top-K

`build_topk_confusion` monta uma matriz classe-da-consulta x classe-dos-vizinhos,
somando os vizinhos recuperados e normalizando por linha. A diagonal representa
acertos; valores fora da diagonal sao confusoes.

### 3.4 Falsos positivos

`analyze_false_positives` filtra as consultas das classes criticas e conta, entre os
vizinhos, quais pertencem a outras classes (falsos positivos), guardando exemplos.

### 3.5 Graficos e montagens

- `plot_pr_curve`: curva Precision (y) x Recall (x) anotada por K.
- `plot_confusion`: heatmap da matriz de confusao Top-K.
- `make_montage`: cria uma tira com a imagem de consulta + Top-K, com borda
  azul (consulta), verde (acerto) ou vermelha (falso positivo). Usa `file_path`
  local; em modo AWS (`s3://`) as montagens sao puladas.

### 3.6 Relatorio

`write_report` escreve `docs/issue5_evaluation.md` com: tabela Precision/Recall,
curva PR, matriz de confusao, analise de falsos positivos por classe critica,
exemplos visuais e um resumo para a secao de Resultados.

## 4. Dependencias

- `matplotlib` (graficos), `numpy` (agregacao), `Pillow` (montagens),
  `psycopg2` (banco). Todas em `requirements.txt`.

## 5. Observacoes

- Requer o banco carregado e o Postgres no ar (porta 5433 por padrao).
- Nao le nomes de arquivo para julgar semelhanca: usa apenas os embeddings.
- E seguro re-executar; sobrescreve relatorio e imagens de saida.
