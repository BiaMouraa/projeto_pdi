# Issue 5 - Validacao de Casos Criticos e Avaliacao de Desempenho

Este documento descreve a entrega da Issue 5 e como reproduzi-la. O script principal
e [`retrieval_evaluation.py`](../retrieval_evaluation.py) (detalhado em
[retrieval_evaluation.md](retrieval_evaluation.md)).

## Objetivo

Com a busca semantica funcionando (Issue 4), a Issue 5 mede **quao boa** e a
recuperacao e investiga as **classes mais dificeis**:

1. Precision e Recall gerais do sistema de recuperacao.
2. Baterias de teste focadas nos casos criticos (Bearded Iris / Douglas Iris).
3. Analise qualitativa dos falsos positivos entre classes semelhantes.
4. Graficos de desempenho (curva Precision-Recall e matriz de confusao Top-K).
5. Consolidacao de dados + imagens para a secao de Resultados e Discussao.

## Ressalva importante sobre as classes criticas

O dataset usado no pipeline (`Oxford Flower 17`) possui **uma unica classe `iris`**.
As especies **Bearded Iris** e **Douglas Iris** citadas no artigo base pertencem a um
dataset de iris mais granular (varias especies do genero Iris).

Consequencia pratica:

- Nao e possivel separar Bearded vs Douglas apenas com o Oxford Flower 17.
- O script trata a classe `iris` como **caso critico padrao** e analisa com quais
  outras flores ela e confundida no espaco latente.
- Se, futuramente, um dataset com essas especies for carregado, basta rodar com
  `--critical-classes "bearded iris" "douglas iris"` que a mesma rotina funciona.

## Conceitos de metricas

### Precision@K

Dos K vizinhos retornados, qual fracao pertence a mesma classe da consulta.

```
Precision@K = (vizinhos da mesma classe) / K
```

### Recall@K

Dos itens relevantes existentes no banco (mesma classe, excluindo a propria consulta),
qual fracao apareceu no Top-K.

```
Recall@K = (vizinhos da mesma classe no Top-K) / (total de imagens da classe - 1)
```

### Macro vs Micro

- **Macro**: calcula a metrica por classe e tira a media das classes (todas pesam igual).
- **Micro**: media sobre todas as consultas (classes maiores pesam mais).

### Matriz de confusao adaptada para Top-K

Cada linha e a classe da consulta; cada coluna e a classe dos vizinhos recuperados.
A celula guarda a fracao dos vizinhos daquela classe. A diagonal alta = boa recuperacao.

### Falso positivo

Vizinho retornado no Top-K que pertence a uma classe **diferente** da consulta.

## Como executar

Pre-requisitos: banco carregado (`load_embeddings.py`) e Postgres no ar (porta 5433).

```bash
python retrieval_evaluation.py --metric cosseno --max-k 10 --report-k 5 10
```

Parametros uteis:

- `--metric`: `cosseno` (padrao) ou `euclidiana`.
- `--max-k`: K maximo (curva PR vai de 1 ate esse valor).
- `--report-k`: K destacados na tabela (padrao `5 10`).
- `--critical-classes`: lista de classes criticas (padrao: classes com "iris").
- `--montage-examples`: quantas montagens por classe critica (padrao 3).

## Saidas geradas

- `docs/issue5_evaluation.md`: relatorio consolidado (tabelas + graficos + FPs).
- `docs/issue5_assets/pr_curve.png`: curva Precision-Recall.
- `docs/issue5_assets/confusion_topk.png`: matriz de confusao Top-K.
- `docs/issue5_assets/montage_*.png`: exemplos visuais (query + vizinhos, FP em vermelho).

## Checklist da Issue 5

| Tarefa | Onde |
|--------|------|
| Precision/Recall geral | `retrieval_evaluation.py` secao 1 do relatorio |
| Baterias nos casos criticos | `--critical-classes` / deteccao automatica de `iris` |
| Analise de falsos positivos | secao 4 do relatorio |
| Graficos (PR + confusao Top-K) | `pr_curve.png`, `confusion_topk.png` |
| Consolidacao p/ Resultados | `docs/issue5_evaluation.md` + montagens |
