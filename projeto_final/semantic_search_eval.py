import argparse
import datetime as dt
import statistics
import time
from pathlib import Path

import psycopg2
from botocore.exceptions import ClientError, NoCredentialsError
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from pipeline_config import DB_CONFIG as DEFAULT_DB_CONFIG
from pipeline_config import (
    AWS_IMAGE_BUCKET,
    download_image_to_temp,
    is_aws_mode,
    normalize_image_s3_uri,
    parse_s3_uri,
)

OPERATOR_BY_METRIC = {
    "euclidiana": "<->",
    "cosseno": "<=>",
}

TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera embedding de uma imagem de teste e compara busca semantica "
            "com distancia euclidiana (<->) e cosseno (<=>)."
        )
    )
    image_group = parser.add_mutually_exclusive_group(required=False)
    image_group.add_argument(
        "--image",
        help=(
            "Caminho local ou URI s3://... da imagem de teste. "
            f"Em PIPELINE_MODE=aws, URIs sao resolvidas em s3://{AWS_IMAGE_BUCKET}/."
        ),
    )
    image_group.add_argument(
        "--image-id",
        help="image_id cadastrado em flower_embeddings; o script usa file_path do banco.",
    )
    parser.add_argument(
        "--list-db-samples",
        action="store_true",
        help="Lista image_id, class_name e file_path no banco e encerra.",
    )
    parser.add_argument(
        "--local-data-root",
        default="local_data",
        help=(
            "Pasta local que espelha as chaves S3 (ex.: local_data/processed/rose/foto.jpg). "
            "Usada antes de baixar do S3 (ignorada se PIPELINE_MODE=aws)."
        ),
    )
    parser.add_argument(
        "--query-class",
        default=None,
        help="Classe real da imagem de teste (opcional, para medir coerencia).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Quantidade de repeticoes para benchmark de tempo.",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Valores de K para retorno dos vizinhos mais proximos.",
    )
    parser.add_argument("--db-name", default=DEFAULT_DB_CONFIG["dbname"])
    parser.add_argument("--db-user", default=DEFAULT_DB_CONFIG["user"])
    parser.add_argument("--db-password", default=DEFAULT_DB_CONFIG["password"])
    parser.add_argument("--db-host", default=DEFAULT_DB_CONFIG["host"])
    parser.add_argument("--db-port", default=DEFAULT_DB_CONFIG["port"])
    parser.add_argument(
        "--report-path",
        default="docs/issue4_metric_evaluation.md",
        help="Arquivo markdown de saida com o resultado consolidado.",
    )
    return parser.parse_args()


def load_feature_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    return model, device


def lookup_image_in_db(db_config, image_id):
    sql = """
    SELECT class_name, file_path
    FROM flower_embeddings
    WHERE image_id = %s
    LIMIT 1;
    """
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (image_id,))
            row = cursor.fetchone()
    if not row:
        raise ValueError(f"Nenhum registro encontrado para image_id={image_id!r}.")
    return row[0], row[1]


def list_db_samples(db_config, limit=10):
    sql = """
    SELECT image_id, class_name, file_path
    FROM flower_embeddings
    ORDER BY id
    LIMIT %s;
    """
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (limit,))
            return cursor.fetchall()


def local_mirror_for_s3(s3_uri, local_data_root):
    _bucket, key = parse_s3_uri(s3_uri)
    root = Path(local_data_root).expanduser().resolve()
    return root / key


def resolve_image_path(image_arg, local_data_root):
    if image_arg.startswith("s3://"):
        # Modo aws: sempre resolve no bucket oficial iris-cv-latente-data.
        uri = normalize_image_s3_uri(image_arg) if is_aws_mode() else image_arg
        if is_aws_mode():
            try:
                temp_path = download_image_to_temp(uri)
                return temp_path, uri, True
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    raise FileNotFoundError(
                        f"Objeto nao encontrado no S3: {uri}"
                    ) from exc
                raise
            except NoCredentialsError as exc:
                raise RuntimeError(
                    "Credenciais AWS nao configuradas (NoCredentialsError).\n"
                    f"PIPELINE_MODE=aws exige acesso ao bucket {AWS_IMAGE_BUCKET}.\n"
                    f"URI tentada: {uri}"
                ) from exc

        mirror = local_mirror_for_s3(uri, local_data_root)
        if mirror.is_file():
            return mirror, uri, False

        try:
            temp_path = download_image_to_temp(uri)
            return temp_path, uri, True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"Objeto nao encontrado no S3: {uri}"
                ) from exc
            raise
        except NoCredentialsError as exc:
            raise RuntimeError(
                "Credenciais AWS nao configuradas (NoCredentialsError).\n"
                f"Opcoes:\n"
                f"  1) Copie a imagem para o espelho local: {mirror}\n"
                f"  2) Use --image com caminho absoluto de um .jpg no disco\n"
                f"  3) Configure AWS (aws configure ou variaveis AWS_ACCESS_KEY_ID/SECRET)\n"
                f"URI tentada: {uri}"
            ) from exc

    path = Path(image_arg).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        if is_aws_mode():
            hint = (
                f"Em PIPELINE_MODE=aws as imagens ficam em s3://{AWS_IMAGE_BUCKET}/. "
                "Use --image s3://... ou --image-id."
            )
        else:
            hint = (
                "Imagens do pipeline ficam no S3; sem AWS, use --image com um .jpg local "
                f"ou espelhe a chave S3 em --local-data-root (padrao: {local_data_root!r})."
            )
        raise FileNotFoundError(f"Imagem nao encontrada: {path}\n{hint}")
    return path, str(path), False


def build_embedding(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = model(tensor).squeeze(0).cpu().numpy().tolist()
    return vector


def to_vector_literal(embedding):
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"


def run_query(conn, vector_literal, operator, top_k):
    sql = f"""
    SELECT
        image_id,
        class_name,
        file_path,
        embedding {operator} %s::vector AS score
    FROM flower_embeddings
    ORDER BY embedding {operator} %s::vector
    LIMIT %s;
    """
    start = time.perf_counter()
    with conn.cursor() as cursor:
        cursor.execute(sql, (vector_literal, vector_literal, top_k))
        rows = cursor.fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return rows, elapsed_ms


def coherence_at_k(rows, query_class):
    if not query_class:
        return None
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row[1] == query_class)
    return hits / len(rows)


def evaluate_metric(conn, metric, vector_literal, top_ks, runs, query_class):
    operator = OPERATOR_BY_METRIC[metric]
    metric_report = {"metric": metric, "operator": operator, "top_k": {}}

    for k in top_ks:
        latencies = []
        sample_rows = []

        for run_idx in range(runs):
            rows, elapsed_ms = run_query(conn, vector_literal, operator, k)
            latencies.append(elapsed_ms)
            if run_idx == 0:
                sample_rows = rows

        metric_report["top_k"][k] = {
            "avg_ms": statistics.mean(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "coherence": coherence_at_k(sample_rows, query_class),
            "results": sample_rows,
        }

    return metric_report


def choose_metric(reports):
    def avg_latency(report):
        return statistics.mean(v["avg_ms"] for v in report["top_k"].values())

    def avg_coherence(report):
        values = [v["coherence"] for v in report["top_k"].values() if v["coherence"] is not None]
        return statistics.mean(values) if values else None

    euclidean = reports["euclidiana"]
    cosine = reports["cosseno"]
    e_coh = avg_coherence(euclidean)
    c_coh = avg_coherence(cosine)
    e_lat = avg_latency(euclidean)
    c_lat = avg_latency(cosine)

    if e_coh is not None and c_coh is not None:
        if c_coh > e_coh:
            return "cosseno", "Maior coerencia visual media."
        if e_coh > c_coh:
            return "euclidiana", "Maior coerencia visual media."
        if c_lat <= e_lat:
            return "cosseno", "Empate em coerencia e menor latencia media."
        return "euclidiana", "Empate em coerencia e menor latencia media."

    if c_lat <= e_lat:
        return "cosseno", "Sem rotulo de classe para coerencia; menor latencia media."
    return "euclidiana", "Sem rotulo de classe para coerencia; menor latencia media."


def print_report_to_console(report):
    metric = report["metric"]
    operator = report["operator"]
    print(f"\n=== Metrica: {metric} (operador {operator}) ===")

    for k, data in report["top_k"].items():
        print(
            f"\nTop-{k} | tempo medio: {data['avg_ms']:.2f} ms "
            f"(min {data['min_ms']:.2f} ms / max {data['max_ms']:.2f} ms)"
        )
        if data["coherence"] is not None:
            print(f"Coerencia visual (mesma classe): {data['coherence']:.2%}")
        for idx, row in enumerate(data["results"], start=1):
            image_id, class_name, file_path, score = row
            print(f"{idx:02d}. {image_id} | classe={class_name} | score={score:.6f} | {file_path}")


def build_markdown_report(image_path, query_class, runs, reports, selected_metric, reason):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Issue 4 - Avaliacao das metricas de busca semantica",
        "",
        f"- Data da execucao: {now}",
        f"- Imagem de teste: `{image_path}`",
        f"- Classe de referencia para coerencia: `{query_class}`" if query_class else "- Classe de referencia para coerencia: nao informada",
        f"- Repeticoes de benchmark por consulta: `{runs}`",
        "",
        "## Consultas SQL implementadas",
        "",
        "- Distancia Euclidiana (`<->`):",
        "```sql",
        "SELECT image_id, class_name, file_path, embedding <-> $1::vector AS score",
        "FROM flower_embeddings",
        "ORDER BY embedding <-> $1::vector",
        "LIMIT $2;",
        "```",
        "",
        "- Distancia de Cosseno (`<=>`):",
        "```sql",
        "SELECT image_id, class_name, file_path, embedding <=> $1::vector AS score",
        "FROM flower_embeddings",
        "ORDER BY embedding <=> $1::vector",
        "LIMIT $2;",
        "```",
        "",
        "## Resultado comparativo",
        "",
    ]

    for metric_name in ("euclidiana", "cosseno"):
        metric = reports[metric_name]
        lines.append(f"### Metrica `{metric_name}` (operador `{metric['operator']}`)")
        for k, data in metric["top_k"].items():
            lines.append(f"- Top-{k}: tempo medio `{data['avg_ms']:.2f} ms` (min `{data['min_ms']:.2f}` / max `{data['max_ms']:.2f}`)")
            if data["coherence"] is not None:
                lines.append(f"- Top-{k}: coerencia visual `{data['coherence']:.2%}`")
            lines.append(f"- Top-{k}: primeiros resultados")
            for idx, row in enumerate(data["results"], start=1):
                image_id, class_name, file_path, score = row
                lines.append(
                    f"  - {idx:02d}. `{image_id}` | classe `{class_name}` | score `{score:.6f}` | `{file_path}`"
                )
        lines.append("")

    lines.extend(
        [
            "## Metrica definitiva recomendada",
            "",
            f"- Metrica escolhida: `{selected_metric}`",
            f"- Justificativa: {reason}",
            "",
            "## Observacao",
            "",
            "- Se necessario, repita com outras imagens de teste para validar estabilidade da escolha.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()

    db_config = {
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
        "host": args.db_host,
        "port": args.db_port,
    }

    if args.list_db_samples:
        rows = list_db_samples(db_config)
        if not rows:
            print("Nenhum registro em flower_embeddings. Rode load_embeddings.py antes.")
            return
        print(
            "Amostras em flower_embeddings "
            f"(use --image-id; em aws as imagens estao em s3://{AWS_IMAGE_BUCKET}/):"
        )
        for image_id, class_name, file_path in rows:
            print(f"  {image_id}\t{class_name}\t{file_path}")
        return

    if not args.image and not args.image_id:
        raise SystemExit(
            "Informe --image, --image-id ou --list-db-samples. "
            "Ex.: python semantic_search_eval.py --list-db-samples"
        )

    top_ks = sorted(set(args.top_k))
    if any(k <= 0 for k in top_ks):
        raise ValueError("Todos os valores de --top-k devem ser maiores que zero.")
    if args.runs <= 0:
        raise ValueError("--runs deve ser maior que zero.")

    query_class = args.query_class
    image_label = None
    temp_download = None

    if args.image_id:
        class_from_db, file_ref = lookup_image_in_db(db_config, args.image_id)
        query_class = query_class or class_from_db
        image_path, image_label, is_temp = resolve_image_path(
            file_ref, args.local_data_root
        )
        if is_temp:
            temp_download = image_path
    else:
        image_path, image_label, is_temp = resolve_image_path(
            args.image, args.local_data_root
        )
        if is_temp:
            temp_download = image_path

    print("Carregando modelo de extracao de features...")
    model, device = load_feature_extractor()

    print("Gerando embedding da imagem de teste...")
    embedding = build_embedding(image_path, model, device)
    vector_literal = to_vector_literal(embedding)

    print("Conectando ao banco e executando benchmarks...")
    with psycopg2.connect(**db_config) as conn:
        reports = {
            "euclidiana": evaluate_metric(
                conn=conn,
                metric="euclidiana",
                vector_literal=vector_literal,
                top_ks=top_ks,
                runs=args.runs,
                query_class=query_class,
            ),
            "cosseno": evaluate_metric(
                conn=conn,
                metric="cosseno",
                vector_literal=vector_literal,
                top_ks=top_ks,
                runs=args.runs,
                query_class=query_class,
            ),
        }

    for report in reports.values():
        print_report_to_console(report)

    selected_metric, reason = choose_metric(reports)
    print("\n=== Recomendacao Final ===")
    print(f"Metrica escolhida: {selected_metric}")
    print(f"Justificativa: {reason}")

    report_md = build_markdown_report(
        image_path=image_label,
        query_class=query_class,
        runs=args.runs,
        reports=reports,
        selected_metric=selected_metric,
        reason=reason,
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Relatorio salvo em: {report_path}")

    if temp_download is not None:
        temp_download.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
