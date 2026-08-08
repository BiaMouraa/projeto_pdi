#!/usr/bin/env bash
# Pipeline offline (Mac). Execute: bash run_offline_pipeline.sh
# Rode UM bloco por vez se preferir; nao cole comentarios (#) na mesma linha dos comandos.

set -euo pipefail
cd "$(dirname "$0")"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> 1/5 preprocess (modo local)"
python preprocess.py --mode local

echo "==> 2/5 extractor"
python extractor.py --mode local

echo "==> 3/5 Postgres (Docker)"
if ! docker info >/dev/null 2>&1; then
  echo "ERRO: Docker nao esta rodando. Abra o Docker Desktop e execute de novo:"
  echo "  docker compose up -d"
  exit 1
fi
docker compose up -d
echo "Aguardando Postgres..."
sleep 5

echo "==> 4/5 load_embeddings"
python load_embeddings.py --mode local

echo "==> 5/5 Issue 4 (semantic search)"
SAMPLE="$(find data/processed -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 1)"
if [[ -z "${SAMPLE}" ]]; then
  echo "ERRO: nenhuma imagem em data/processed"
  exit 1
fi
CLASS="$(basename "$(dirname "${SAMPLE}")")"
echo "Imagem de teste: ${SAMPLE} (classe: ${CLASS})"

python semantic_search_eval.py --list-db-samples
python semantic_search_eval.py \
  --image "${SAMPLE}" \
  --query-class "${CLASS}" \
  --top-k 5 10 \
  --runs 5

echo "Concluido. Relatorio: docs/issue4_metric_evaluation.md"
