"""
Valida os resultados gerados contra os requisitos do trabalho de PDI.

Uso:
    python validar_resultados.py
    python validar_resultados.py --imagem GrayAndMagenta
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ARQUIVOS_MODIFICADO = [
    "1_magnitude_di_zenzo.png",
    "2_nms_di_zenzo.png",
    "3_borda_final_di_zenzo.png",
    "7_gabor_escalar_magnitude.png",
    "8_gabor_escalar_nms.png",
    "9_gabor_escalar_borda_final.png",
    "config_usado.json",
]

ARQUIVOS_TRADICIONAL = [
    "4_tradicional_magnitude.png",
    "5_tradicional_nms.png",
    "6_tradicional_borda_final.png",
    "config_usado.json",
]


def listar_imagens(pasta):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in Path(pasta).iterdir() if p.suffix.lower() in exts)


def contar_pixels_borda(caminho):
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return int(np.sum(img > 127))


def medir_espessura_borda(caminho, amostras=5):
    """Estima espessura média da borda em pixels (ideal ~1)."""
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    espessuras = []
    coords = np.argwhere(img > 127)

    if len(coords) == 0:
        return 0.0

    rng = np.random.default_rng(42)
    indices = rng.choice(len(coords), size=min(amostras, len(coords)), replace=False)

    for idx in indices:
        y, x = coords[idx]
        # mede espessura na direção horizontal
        esq = x
        while esq > 0 and img[y, esq] > 127:
            esq -= 1
        dir_ = x
        while dir_ < w - 1 and img[y, dir_] > 127:
            dir_ += 1
        espessuras.append(dir_ - esq - 1)

    return float(np.mean(espessuras)) if espessuras else 0.0


def validar_pasta_modificada(pasta):
    erros = []
    for arq in ARQUIVOS_MODIFICADO:
        if not os.path.exists(os.path.join(pasta, arq)):
            erros.append(f"Faltando: {arq}")
    return erros


def validar_pasta_tradicional(pasta):
    erros = []
    for arq in ARQUIVOS_TRADICIONAL:
        if not os.path.exists(os.path.join(pasta, arq)):
            erros.append(f"Faltando: {arq}")
    return erros


def encontrar_pasta_baseline(nome_imagem):
    base = Path("results") / nome_imagem
    if not base.is_dir():
        return None
    candidatos = sorted(
        p for p in base.iterdir()
        if p.is_dir() and "baseline" in p.name and (p / "1_magnitude_di_zenzo.png").exists()
    )
    return candidatos[0] if candidatos else None


def main():
    parser = argparse.ArgumentParser(description="Valida resultados do pipeline PDI.")
    parser.add_argument("--imagem", default=None, help="Validar só uma imagem (ex: GrayAndMagenta)")
    args = parser.parse_args()

    if not os.path.isdir("images"):
        print("[ERRO] Pasta images/ não encontrada.")
        return

    imagens = listar_imagens("images")
    if args.imagem:
        imagens = [p for p in imagens if p.stem == args.imagem]

    if not imagens:
        print("[ERRO] Nenhuma imagem encontrada.")
        return

    print("=" * 60)
    print("VALIDAÇÃO DE RESULTADOS — PDI 2026.1")
    print("=" * 60)

    total_ok = 0
    total_erro = 0
    relatorio = []

    for img_path in imagens:
        nome = img_path.stem
        print(f"\n--- {nome} ---")
        item = {"imagem": nome, "checks": []}

        pasta_trad = Path("results") / nome / "canny_tradicional"
        if pasta_trad.is_dir():
            erros = validar_pasta_tradicional(pasta_trad)
            if erros:
                print(f"  [FALHA] canny_tradicional: {erros}")
                item["checks"].append({"tradicional": "FALHA", "erros": erros})
                total_erro += 1
            else:
                px_trad = contar_pixels_borda(pasta_trad / "6_tradicional_borda_final.png")
                esp_trad = medir_espessura_borda(pasta_trad / "6_tradicional_borda_final.png")
                print(f"  [OK] canny_tradicional — pixels borda: {px_trad}, espessura ~{esp_trad:.1f}px")
                item["checks"].append({"tradicional": "OK", "pixels": px_trad, "espessura": esp_trad})
                total_ok += 1
        else:
            print("  [FALHA] canny_tradicional/ não existe")
            item["checks"].append({"tradicional": "AUSENTE"})
            total_erro += 1

        pasta_base = encontrar_pasta_baseline(nome)
        if pasta_base:
            erros = validar_pasta_modificada(pasta_base)
            if erros:
                print(f"  [FALHA] baseline: {erros}")
                item["checks"].append({"baseline": "FALHA", "erros": erros})
                total_erro += 1
            else:
                px_mod = contar_pixels_borda(pasta_base / "3_borda_final_di_zenzo.png")
                px_esc = contar_pixels_borda(pasta_base / "9_gabor_escalar_borda_final.png")
                esp_mod = medir_espessura_borda(pasta_base / "3_borda_final_di_zenzo.png")
                print(f"  [OK] baseline modificado — pixels: {px_mod}, espessura ~{esp_mod:.1f}px")
                print(f"  [OK] baseline Gabor escalar — pixels: {px_esc}")
                item["checks"].append({
                    "baseline": "OK",
                    "pixels_di_zenzo": px_mod,
                    "pixels_escalar": px_esc,
                    "espessura_di_zenzo": esp_mod,
                })
                total_ok += 1
        else:
            print("  [FALHA] pasta baseline não encontrada (rode run_experimentos.py)")
            item["checks"].append({"baseline": "AUSENTE"})
            total_erro += 1

        relatorio.append(item)

    # Teste crítico GrayAndMagenta
    print("\n" + "=" * 60)
    print("TESTE CRÍTICO: GrayAndMagenta (borda cromática)")
    print("=" * 60)
    gm_trad = Path("results/GrayAndMagenta/canny_tradicional/6_tradicional_borda_final.png")
    gm_base = encontrar_pasta_baseline("GrayAndMagenta")

    if gm_trad.exists() and gm_base:
        px_t = contar_pixels_borda(gm_trad)
        px_m = contar_pixels_borda(gm_base / "3_borda_final_di_zenzo.png")
        print(f"  Tradicional (Sobel):  {px_t} pixels de borda")
        print(f"  Modificado (Di Zenzo): {px_m} pixels de borda")
        if px_m > px_t:
            print("  [OK] Modificado detectou MAIS bordas que tradicional (esperado para borda cromática)")
        elif px_m == px_t == 0:
            print("  [ATENÇÃO] Nenhum detectou bordas — revisar limiares (t_low/t_high)")
        else:
            print("  [ATENÇÃO] Verificar visualmente — modificado deveria ver borda cinza/magenta")
    else:
        print("  [PENDENTE] Rode os testes primeiro")

    # Contagem de experimentos
    print("\n" + "=" * 60)
    print("CONTAGEM DE EXPERIMENTOS")
    print("=" * 60)
    for img_path in imagens:
        nome = img_path.stem
        pasta = Path("results") / nome
        if pasta.is_dir():
            runs = [d.name for d in pasta.iterdir() if d.is_dir() and d.name != "canny_tradicional"]
            print(f"  {nome}: {len(runs)} execuções modificado")

    out_path = "results/validacao_relatorio.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    print(f"\nRelatório JSON salvo em: {out_path}")
    print(f"Resumo: {total_ok} checks OK, {total_erro} checks com problema")


if __name__ == "__main__":
    main()
