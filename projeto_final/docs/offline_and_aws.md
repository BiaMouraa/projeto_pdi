# Pipeline offline (Mac) vs AWS (EC2)

Use este guia para **fechar a Issue 4 agora no Mac** e, depois, repetir o fluxo **na VM EC2** quando o outro integrante subir a infra.

## Objetivo do projeto (lembrete)

1. Preprocessar flores (Oxford Flower 17)  
2. Extrair embeddings (MobileNetV2, 1280 dims)  
3. Carregar no PostgreSQL + `pgvector`  
4. **Issue 4:** buscar vizinhos (Top-5/Top-10), comparar `<->` vs `<=>`, documentar metrica final  

---

## Modos

| Modo | Onde | Dados | Comando base |
|------|------|-------|----------------|
| **`local`** | Mac, sem AWS | `data/processed/`, `data/embeddings.csv` | `--mode local` |
| **`aws`** | EC2 com IAM | S3 `iris-cv-latente-data` | `--mode aws` ou `export PIPELINE_MODE=aws` |

Config central: `pipeline_config.py` (variavel `PIPELINE_MODE` opcional).

---

## AGORA no Mac (offline) — ordem exata

**Importante:** execute **um comando por linha**. Nao cole textos com `# comentario` na mesma linha — o terminal trata `#` como comentario e quebra o comando (como nos erros `unrecognized arguments: # 2` e `no such service: #`).

Alternativa automatica (recomendada):

```bash
cd projeto_final
source .venv/bin/activate
bash run_offline_pipeline.sh
```

Ou passo a passo manual:

Diretorio: `projeto_final/`

```bash
python preprocess.py --mode local
```

```bash
python extractor.py --mode local
```

Abra o **Docker Desktop**, depois:

```bash
docker compose up -d
```

```bash
python load_embeddings.py --mode local
```

Escolha uma imagem real (nao use `ROSE/arquivo.jpg` — sao placeholders):

```bash
find data/processed -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | head -3
```

```bash
python semantic_search_eval.py --list-db-samples
python semantic_search_eval.py --image data/processed/NOME_DA_PASTA/foto.jpg --query-class NOME_DA_PASTA --top-k 5 10 --runs 5
```

Substitua `NOME_DA_PASTA` e `foto.jpg` pela saida do `find`.

**Entregavel da Issue 4:** arquivo gerado `docs/issue4_metric_evaluation.md` (tempo + coerencia + metrica escolhida).

### Checklist antes do push

- [ ] `data/embeddings.csv` existe e tem linhas  
- [ ] `docker compose ps` mostra Postgres healthy  
- [ ] `semantic_search_eval.py --list-db-samples` lista registros  
- [ ] Relatorio `docs/issue4_metric_evaluation.md` gerado  
- [ ] **Nao commitar** `data/` (esta no `.gitignore`)  

---

## DEPOIS na EC2 (AWS)

Na instancia (`iris-worker`), com role S3:

```bash
export PIPELINE_MODE=aws
cd ~/projeto_pdi/projeto_final
source .venv/bin/activate   # criar venv la se necessario

python preprocess.py --mode aws
python extractor.py --mode aws
# Postgres: docker compose up -d (instalar Docker na EC2 se ainda nao tiver)
python load_embeddings.py --mode aws

python semantic_search_eval.py --list-db-samples
python semantic_search_eval.py --image-id ALGUMA_FOTO.jpg --top-k 5 10 --runs 5
```

Na EC2, `--image-id` funciona com `file_path` S3 se credenciais IAM estiverem ativas.

---

## O que subir no Git (integracao)

Arquivos da entrega (Issue 4 + dual mode):

- `pipeline_config.py`
- `preprocess.py`, `extractor.py`, `load_embeddings.py` (modos local/aws)
- `semantic_search_eval.py`, `sql/semantic_search_queries.sql`
- `docs/issue4_semantic_search.md`, `docs/offline_and_aws.md`
- `docs/issue4_metric_evaluation.md` (apos rodar o benchmark)
- `README.md` atualizado

O outro integrante so precisa **nao sobrescrever** esses arquivos ao subir VM; na EC2 roda `--mode aws`.

---

## Problemas comuns

| Sintoma | Causa | Acao |
|---------|--------|------|
| `NoCredentialsError` no Mac | Tentou S3 sem AWS | Use `--mode local` |
| Imagem nao encontrada | Caminho errado | Use arquivo dentro de `data/processed/...` |
| Postgres connection refused | Docker parado | `docker compose up -d` |
| Kaggle download falha | Credencial Kaggle | Configurar kagglehub/kaggle API no Mac |
