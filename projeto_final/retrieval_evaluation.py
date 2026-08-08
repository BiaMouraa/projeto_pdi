"""Issue 5 - Validacao de casos criticos e avaliacao de desempenho.

Calcula Precision/Recall do sistema de recuperacao (busca Top-K no pgvector),
foca em classes criticas, analisa falsos positivos entre classes semelhantes e
gera graficos (curva Precision-Recall e matriz de confusao adaptada para Top-K).

Uso tipico (offline, apos load_embeddings.py):

    python retrieval_evaluation.py --metric cosseno --max-k 10

Saidas:
    - docs/issue5_evaluation.md            (relatorio consolidado)
    - docs/issue5_assets/pr_curve.png      (curva Precision-Recall)
    - docs/issue5_assets/confusion_topk.png(matriz de confusao Top-K)
    - docs/issue5_assets/montage_*.png     (exemplos query + vizinhos)
"""

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psycopg2

from pipeline_config import DB_CONFIG as DEFAULT_DB_CONFIG
from pipeline_config import download_image_bytes, is_aws_mode, normalize_image_s3_uri

OPERATOR_BY_METRIC = {
    "euclidiana": "<->",
    "cosseno": "<=>",
}


# ==========================================
# Argumentos de linha de comando
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Issue 5: Precision/Recall, casos criticos, falsos positivos e "
            "graficos de desempenho para a busca semantica Top-K."
        )
    )
    parser.add_argument(
        "--metric",
        choices=tuple(OPERATOR_BY_METRIC.keys()),
        default="cosseno",
        help="Metrica de distancia usada na recuperacao (padrao: cosseno).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="K maximo para os vizinhos e para a curva Precision-Recall.",
    )
    parser.add_argument(
        "--report-k",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Valores de K destacados no relatorio (padrao: 5 10).",
    )
    parser.add_argument(
        "--critical-classes",
        nargs="+",
        default=None,
        help=(
            "Classes criticas a investigar. Se omitido, detecta classes que "
            "contenham 'iris'. No Oxford Flower 17 existe apenas a classe 'iris'."
        ),
    )
    parser.add_argument(
        "--montage-examples",
        type=int,
        default=3,
        help="Quantidade de exemplos (query + vizinhos) por classe critica.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/issue5_assets",
        help="Pasta para salvar graficos e montagens.",
    )
    parser.add_argument(
        "--report-path",
        default="docs/issue5_evaluation.md",
        help="Arquivo markdown do relatorio consolidado.",
    )
    parser.add_argument("--db-name", default=DEFAULT_DB_CONFIG["dbname"])
    parser.add_argument("--db-user", default=DEFAULT_DB_CONFIG["user"])
    parser.add_argument("--db-password", default=DEFAULT_DB_CONFIG["password"])
    parser.add_argument("--db-host", default=DEFAULT_DB_CONFIG["host"])
    parser.add_argument("--db-port", default=DEFAULT_DB_CONFIG["port"])
    return parser.parse_args()


# ==========================================
# Acesso ao banco
# ==========================================
def get_class_counts(conn):
    sql = "SELECT class_name, COUNT(*) FROM flower_embeddings GROUP BY class_name ORDER BY class_name;"
    with conn.cursor() as cursor:
        cursor.execute(sql)
        return dict(cursor.fetchall())


def fetch_topk_all(conn, operator, max_k):
    """Retorna, para cada imagem, seus max_k vizinhos mais proximos (exclui a si mesma).

    Usa CROSS JOIN LATERAL para calcular tudo em uma unica consulta, aproveitando
    o indice HNSW da coluna embedding.
    """
    sql = f"""
    SELECT
        q.id          AS q_id,
        q.image_id    AS q_image,
        q.class_name  AS q_class,
        q.file_path   AS q_path,
        n.image_id    AS n_image,
        n.class_name  AS n_class,
        n.file_path   AS n_path,
        n.score       AS score
    FROM flower_embeddings q
    CROSS JOIN LATERAL (
        SELECT image_id, class_name, file_path,
               embedding {operator} q.embedding AS score
        FROM flower_embeddings
        WHERE id <> q.id
        ORDER BY embedding {operator} q.embedding
        LIMIT %s
    ) n
    ORDER BY q.id, n.score;
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (max_k,))
        rows = cursor.fetchall()

    queries = {}
    for q_id, q_image, q_class, q_path, n_image, n_class, n_path, score in rows:
        entry = queries.setdefault(
            q_id,
            {"image_id": q_image, "class_name": q_class, "file_path": q_path, "neighbors": []},
        )
        entry["neighbors"].append(
            {"image_id": n_image, "class_name": n_class, "file_path": n_path, "score": float(score)}
        )
    return queries


# ==========================================
# Metricas Precision / Recall
# ==========================================
def precision_recall_at_k(queries, class_counts, k):
    """Precision@k e Recall@k medios (macro) e por classe."""
    per_class_precision = defaultdict(list)
    per_class_recall = defaultdict(list)

    for entry in queries.values():
        q_class = entry["class_name"]
        topk = entry["neighbors"][:k]
        relevant_retrieved = sum(1 for n in topk if n["class_name"] == q_class)

        precision = relevant_retrieved / k if k > 0 else 0.0
        total_relevant = max(class_counts.get(q_class, 1) - 1, 0)
        recall = min(relevant_retrieved / total_relevant, 1.0) if total_relevant > 0 else 0.0

        per_class_precision[q_class].append(precision)
        per_class_recall[q_class].append(recall)

    class_precision = {c: float(np.mean(v)) for c, v in per_class_precision.items()}
    class_recall = {c: float(np.mean(v)) for c, v in per_class_recall.items()}

    all_precisions = [p for v in per_class_precision.values() for p in v]
    all_recalls = [r for v in per_class_recall.values() for r in v]

    macro_precision = float(np.mean(list(class_precision.values()))) if class_precision else 0.0
    macro_recall = float(np.mean(list(class_recall.values()))) if class_recall else 0.0
    micro_precision = float(np.mean(all_precisions)) if all_precisions else 0.0
    micro_recall = float(np.mean(all_recalls)) if all_recalls else 0.0

    return {
        "k": k,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "class_precision": class_precision,
        "class_recall": class_recall,
    }


def build_pr_curve(queries, class_counts, max_k):
    ks = list(range(1, max_k + 1))
    precisions, recalls = [], []
    for k in ks:
        m = precision_recall_at_k(queries, class_counts, k)
        precisions.append(m["macro_precision"])
        recalls.append(m["macro_recall"])
    return ks, precisions, recalls


# ==========================================
# Matriz de confusao adaptada para Top-K
# ==========================================
def build_topk_confusion(queries, classes, k):
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=float)
    for entry in queries.values():
        q_class = entry["class_name"]
        if q_class not in idx:
            continue
        for n in entry["neighbors"][:k]:
            n_class = n["class_name"]
            if n_class in idx:
                matrix[idx[q_class], idx[n_class]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    return matrix, normalized


# ==========================================
# Analise de falsos positivos
# ==========================================
def analyze_false_positives(queries, critical_classes, k):
    """Para cada classe critica, conta as classes que aparecem como falso positivo."""
    result = {}
    for crit in critical_classes:
        fp_counter = defaultdict(int)
        examples = []
        total_neighbors = 0
        for entry in queries.values():
            if entry["class_name"] != crit:
                continue
            for n in entry["neighbors"][:k]:
                total_neighbors += 1
                if n["class_name"] != crit:
                    fp_counter[n["class_name"]] += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "query_image": entry["image_id"],
                                "query_path": entry["file_path"],
                                "fp_image": n["image_id"],
                                "fp_class": n["class_name"],
                                "fp_path": n["file_path"],
                                "score": n["score"],
                            }
                        )
        result[crit] = {
            "false_positive_by_class": dict(sorted(fp_counter.items(), key=lambda x: -x[1])),
            "total_neighbors": total_neighbors,
            "total_false_positives": sum(fp_counter.values()),
            "examples": examples,
        }
    return result


# ==========================================
# Graficos
# ==========================================
def plot_pr_curve(ks, precisions, recalls, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, marker="o", color="#1f77b4")
    for k, r, p in zip(ks, recalls, precisions):
        ax.annotate(f"K={k}", (r, p), fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Recall (macro)")
    ax.set_ylabel("Precision (macro)")
    ax.set_title("Curva Precision-Recall (variando K)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion(normalized, classes, output_path, k):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Classe recuperada (vizinhos)")
    ax.set_ylabel("Classe da consulta")
    ax.set_title(f"Matriz de confusao Top-{k} (fracao dos vizinhos)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _load_image_safe(path_str, size=(160, 160)):
    from PIL import Image
    import io

    try:
        if path_str.startswith("s3://"):
            uri = normalize_image_s3_uri(path_str) if is_aws_mode() else path_str
            data, _ = download_image_bytes(uri)
            img = Image.open(io.BytesIO(data)).convert("RGB").resize(size)
            return img
        p = Path(path_str)
        if not p.is_file():
            return None
        img = Image.open(p).convert("RGB").resize(size)
        return img
    except Exception:
        return None


def make_montage(entry, k, output_path):
    """Cria uma montagem: imagem de consulta + Top-K, borda verde=acerto, vermelha=FP."""
    from PIL import Image, ImageDraw

    tile = 160
    border = 6
    items = [
        {"path": entry["file_path"], "label": "QUERY", "correct": True, "is_query": True}
    ]
    for n in entry["neighbors"][:k]:
        items.append(
            {
                "path": n["file_path"],
                "label": f"{n['class_name']} ({n['score']:.3f})",
                "correct": n["class_name"] == entry["class_name"],
                "is_query": False,
            }
        )

    loaded = [(_load_image_safe(it["path"], (tile, tile)), it) for it in items]
    if all(img is None for img, _ in loaded):
        return False

    cols = len(loaded)
    width = cols * (tile + 2 * border)
    height = tile + 2 * border + 18
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (img, it) in enumerate(loaded):
        x0 = i * (tile + 2 * border)
        if img is None:
            img = Image.new("RGB", (tile, tile), "#dddddd")
        if it["is_query"]:
            color = "#1f77b4"
        else:
            color = "#2ca02c" if it["correct"] else "#d62728"
        framed = Image.new("RGB", (tile + 2 * border, tile + 2 * border), color)
        framed.paste(img, (border, border))
        canvas.paste(framed, (x0, 0))
        draw.text((x0 + 4, tile + 2 * border + 2), it["label"][:24], fill="black")

    canvas.save(output_path)
    return True


# ==========================================
# Relatorio markdown
# ==========================================
def write_report(path, context):
    lines = []
    a = lines.append
    a("# Issue 5 - Validacao de casos criticos e avaliacao de desempenho")
    a("")
    a(f"- Data da execucao: {context['now']}")
    a(f"- Metrica de recuperacao: `{context['metric']}` (operador `{context['operator']}`)")
    a(f"- K maximo avaliado: `{context['max_k']}`")
    a(f"- Total de imagens (consultas): `{context['total_queries']}`")
    a(f"- Classes criticas investigadas: `{', '.join(context['critical_classes'])}`")
    a("")
    a("> Observacao: o dataset Oxford Flower 17 possui uma unica classe `iris`. As")
    a("> especies **Bearded Iris** e **Douglas Iris** citadas no artigo base vem de um")
    a("> dataset de iris mais granular. Aqui a classe `iris` e tratada como caso critico;")
    a("> use `--critical-classes` se um dataset com essas especies for carregado.")
    a("")

    a("## 1. Precision e Recall gerais")
    a("")
    a("| K | Precision (macro) | Recall (macro) | Precision (micro) | Recall (micro) |")
    a("|---|-------------------|----------------|-------------------|----------------|")
    for m in context["report_metrics"]:
        a(
            f"| {m['k']} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} "
            f"| {m['micro_precision']:.3f} | {m['micro_recall']:.3f} |"
        )
    a("")

    a("## 2. Curva Precision-Recall")
    a("")
    a(f"![Curva Precision-Recall]({context['pr_curve_rel']})")
    a("")

    a("## 3. Matriz de confusao adaptada para Top-K")
    a("")
    a(f"Matriz normalizada por linha em Top-{context['confusion_k']} (fracao dos vizinhos recuperados por classe).")
    a("")
    a(f"![Matriz de confusao Top-K]({context['confusion_rel']})")
    a("")

    a("## 4. Casos criticos e falsos positivos")
    a("")
    for crit, data in context["fp_analysis"].items():
        a(f"### Classe critica: `{crit}` (Top-{context['confusion_k']})")
        a("")
        prec = context["critical_precision"].get(crit)
        rec = context["critical_recall"].get(crit)
        if prec is not None:
            a(f"- Precision media: `{prec:.3f}`")
        if rec is not None:
            a(f"- Recall medio: `{rec:.3f}`")
        a(f"- Vizinhos analisados: `{data['total_neighbors']}`")
        a(f"- Falsos positivos: `{data['total_false_positives']}`")
        a("")
        if data["false_positive_by_class"]:
            a("Classes que mais aparecem como falso positivo:")
            a("")
            a("| Classe confundida | Ocorrencias |")
            a("|-------------------|-------------|")
            for cls, count in data["false_positive_by_class"].items():
                a(f"| {cls} | {count} |")
            a("")
        else:
            a("Nenhum falso positivo encontrado para esta classe.")
            a("")

    a("## 5. Exemplos visuais (query + vizinhos)")
    a("")
    a("Borda azul = consulta, verde = acerto (mesma classe), vermelha = falso positivo.")
    a("")
    if context["montages"]:
        for rel in context["montages"]:
            a(f"![Exemplo de recuperacao]({rel})")
            a("")
    else:
        a("Sem montagens (imagens indisponiveis localmente ou em s3://iris-cv-latente-data).")
        a("")

    a("## 6. Consolidacao para Resultados e Discussao")
    a("")
    a("- Tabela da secao 1 resume Precision/Recall gerais.")
    a("- A curva PR (secao 2) mostra o trade-off ao aumentar K.")
    a("- A matriz de confusao (secao 3) indica com quais classes a busca confunde cada consulta.")
    a("- A secao 4 detalha os falsos positivos das classes criticas.")
    a("- As imagens da secao 5 servem de figura direta para o relatorio.")
    a("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    operator = OPERATOR_BY_METRIC[args.metric]

    db_config = {
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
        "host": args.db_host,
        "port": args.db_port,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Conectando ao banco...")
    with psycopg2.connect(**db_config) as conn:
        class_counts = get_class_counts(conn)
        if not class_counts:
            raise SystemExit("Tabela flower_embeddings vazia. Rode load_embeddings.py antes.")
        classes = sorted(class_counts.keys())

        print(f"Buscando Top-{args.max_k} vizinhos para todas as imagens...")
        queries = fetch_topk_all(conn, operator, args.max_k)

    total_queries = len(queries)
    print(f"{total_queries} consultas avaliadas.")

    # Classes criticas
    if args.critical_classes:
        critical_classes = [c for c in args.critical_classes if c in class_counts]
        missing = set(args.critical_classes) - set(critical_classes)
        for m in missing:
            print(f"Aviso: classe critica '{m}' nao existe no banco e sera ignorada.")
    else:
        critical_classes = [c for c in classes if "iris" in c.lower()]
    if not critical_classes:
        critical_classes = classes[:1]
    print(f"Classes criticas: {critical_classes}")

    # Precision/Recall nos K de relatorio
    report_metrics = [precision_recall_at_k(queries, class_counts, k) for k in sorted(set(args.report_k))]

    # Curva PR
    ks, precisions, recalls = build_pr_curve(queries, class_counts, args.max_k)
    pr_curve_path = output_dir / "pr_curve.png"
    plot_pr_curve(ks, precisions, recalls, pr_curve_path)

    # Matriz de confusao (no maior K de relatorio, limitado ao max_k)
    confusion_k = min(max(args.report_k), args.max_k)
    _, normalized = build_topk_confusion(queries, classes, confusion_k)
    confusion_path = output_dir / "confusion_topk.png"
    plot_confusion(normalized, classes, confusion_path, confusion_k)

    # Falsos positivos das classes criticas
    fp_analysis = analyze_false_positives(queries, critical_classes, confusion_k)

    # Precision/Recall por classe critica no confusion_k
    crit_metrics = precision_recall_at_k(queries, class_counts, confusion_k)
    critical_precision = {c: crit_metrics["class_precision"].get(c) for c in critical_classes}
    critical_recall = {c: crit_metrics["class_recall"].get(c) for c in critical_classes}

    # Montagens de exemplo
    montages = []
    for crit in critical_classes:
        made = 0
        for entry in queries.values():
            if entry["class_name"] != crit:
                continue
            montage_path = output_dir / f"montage_{crit.replace(' ', '_')}_{made+1}.png"
            if make_montage(entry, confusion_k, montage_path):
                montages.append(str(montage_path.relative_to(output_dir.parent)))
                made += 1
            if made >= args.montage_examples:
                break

    context = {
        "now": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric": args.metric,
        "operator": operator,
        "max_k": args.max_k,
        "total_queries": total_queries,
        "critical_classes": critical_classes,
        "report_metrics": report_metrics,
        "pr_curve_rel": str(pr_curve_path.relative_to(output_dir.parent)),
        "confusion_rel": str(confusion_path.relative_to(output_dir.parent)),
        "confusion_k": confusion_k,
        "fp_analysis": fp_analysis,
        "critical_precision": critical_precision,
        "critical_recall": critical_recall,
        "montages": montages,
    }

    write_report(args.report_path, context)

    # Resumo no console
    print("\n=== Resumo Precision/Recall ===")
    for m in report_metrics:
        print(
            f"K={m['k']:>2} | P(macro)={m['macro_precision']:.3f} "
            f"R(macro)={m['macro_recall']:.3f} | "
            f"P(micro)={m['micro_precision']:.3f} R(micro)={m['micro_recall']:.3f}"
        )
    print(f"\nGraficos em: {output_dir}")
    print(f"Relatorio em: {args.report_path}")


if __name__ == "__main__":
    main()
