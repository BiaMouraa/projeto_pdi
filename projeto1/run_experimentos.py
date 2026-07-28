"""
Executa em lote os experimentos obrigatórios do trabalho de PDI.

Uso:
    python run_experimentos.py              # roda tudo conforme experiments_config.json
    python run_experimentos.py --rapido     # só baseline + GrayAndMagenta (teste)
    python run_experimentos.py --exp 1      # só experimento 1 (lambda e sigma)
    python run_experimentos.py --exp 2      # só experimento 2 (histerese)
"""
import argparse
import copy
import json
import os
from pathlib import Path

from projeto1.pipeline import (
    carregar_imagem_rgb,
    executar_canny_tradicional,
    executar_pipeline_modificado,
    gerar_nome_pasta_config,
    pasta_canny_tradicional,
    resolver_pasta_destino,
)


def carregar_json(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def listar_imagens(pasta, extensoes):
    if not os.path.isdir(pasta):
        raise FileNotFoundError(f"Pasta de imagens não encontrada: '{pasta}'")
    imagens = []
    for ext in extensoes:
        imagens.extend(Path(pasta).glob(f"*{ext}"))
    return sorted(set(imagens))


def pasta_ja_existe(nome_imagem, config, sufixo=None):
    nome = gerar_nome_pasta_config(config)
    if sufixo:
        nome = f"{nome}_{sufixo}"
    pasta_dir = os.path.join("results", nome_imagem)
    if not os.path.isdir(pasta_dir):
        return False
    for entry in os.listdir(pasta_dir):
        if entry == nome or entry.startswith(f"{nome}_run"):
            marcador = os.path.join(pasta_dir, entry, "1_magnitude_di_zenzo.png")
            if os.path.exists(marcador):
                return True
    return False


def tradicional_ja_existe(nome_imagem):
    pasta = pasta_canny_tradicional(nome_imagem)
    return os.path.exists(os.path.join(pasta, "6_tradicional_borda_final.png"))


def rodar_modificado(img_rgb, nome_imagem, config, label, exp_cfg, contador):
    if exp_cfg.get('pular_existentes') and pasta_ja_existe(nome_imagem, config, sufixo=label):
        print(f"  [{contador['n']}] PULADO (já existe): {label}")
        contador['n'] += 1
        return

    pasta = resolver_pasta_destino(nome_imagem, config, sufixo=label)
    print(f"  [{contador['n']}] {label} -> {pasta}")
    executar_pipeline_modificado(img_rgb, config, pasta, verbose=False)
    contador['n'] += 1


def rodar_tradicional(img_rgb, nome_imagem, config, exp_cfg, contador):
    if not exp_cfg.get('executar_canny_tradicional', True):
        return
    if exp_cfg.get('pular_existentes') and tradicional_ja_existe(nome_imagem):
        print(f"  [{contador['n']}] PULADO tradicional (já existe): {nome_imagem}")
        contador['n'] += 1
        return
    pasta = pasta_canny_tradicional(nome_imagem)
    print(f"  [{contador['n']}] canny_tradicional -> {pasta}")
    executar_canny_tradicional(img_rgb, config, pasta, verbose=False)
    contador['n'] += 1


def gerar_variacoes(config_base, exp_cfg):
    """Gera lista de (label, config) para todos os experimentos."""
    variacoes = []

    if exp_cfg.get('experimento_baseline', True):
        variacoes.append(("baseline", copy.deepcopy(config_base)))

    for lambda_val in exp_cfg.get('experimento_1_lambda', []):
        cfg = copy.deepcopy(config_base)
        cfg['gabor']['lambda_'] = lambda_val
        variacoes.append((f"exp1_lambda{lambda_val}", cfg))

    for sigma_val in exp_cfg.get('experimento_1_sigma', []):
        cfg = copy.deepcopy(config_base)
        cfg['gabor']['sigma'] = sigma_val
        variacoes.append((f"exp1_sigma{sigma_val}", cfg))

    for i, histerese in enumerate(exp_cfg.get('experimento_2_histerese', []), start=1):
        cfg = copy.deepcopy(config_base)
        cfg['histerese'] = copy.deepcopy(histerese)
        t_low, t_high = histerese['t_low'], histerese['t_high']
        variacoes.append((f"exp2_histerese_tlow{t_low}_thigh{t_high}", cfg))

    return variacoes


def filtrar_variacoes(variacoes, exp_filter):
    if exp_filter == 'all':
        return variacoes
    if exp_filter == '1':
        return [(l, c) for l, c in variacoes if l == 'baseline' or l.startswith('exp1_')]
    if exp_filter == '2':
        return [(l, c) for l, c in variacoes if l == 'baseline' or l.startswith('exp2_')]
    return variacoes


def main():
    parser = argparse.ArgumentParser(description="Executa experimentos em lote para o relatório de PDI.")
    parser.add_argument('--rapido', action='store_true', help='Roda só baseline em GrayAndMagenta (teste).')
    parser.add_argument('--exp', choices=['1', '2', 'all'], default='all', help='Qual experimento rodar.')
    parser.add_argument('--config', default='experiments_config.json', help='Arquivo de config dos experimentos.')
    args = parser.parse_args()

    exp_cfg = carregar_json(args.config)
    config_base = carregar_json(exp_cfg.get('config_base', 'config.json'))

    imagens = listar_imagens(exp_cfg['imagens_dir'], exp_cfg['extensoes'])
    if args.rapido:
        imagens = [p for p in imagens if p.stem == 'GrayAndMagenta'] or imagens[:1]

    if not imagens:
        print(f"Nenhuma imagem encontrada em '{exp_cfg['imagens_dir']}'.")
        return

    variacoes = gerar_variacoes(config_base, exp_cfg)
    variacoes = filtrar_variacoes(variacoes, args.exp)

    total_runs = len(imagens) * len(variacoes)
    print(f"Imagens: {len(imagens)} | Variações: {len(variacoes)} | Total modificado: ~{total_runs}")
    print("-" * 60)

    contador = {'n': 1}

    for img_path in imagens:
        nome = img_path.stem
        print(f"\n=== {img_path.name} ===")
        img_rgb = carregar_imagem_rgb(str(img_path))

        rodar_tradicional(img_rgb, nome, config_base, exp_cfg, contador)

        for label, config in variacoes:
            rodar_modificado(img_rgb, nome, config, label, exp_cfg, contador)

    print("\n" + "=" * 60)
    print(f"[SUCESSO] Batch concluído. Execuções processadas: {contador['n'] - 1}")
    print(f"Resultados em: results/")


if __name__ == "__main__":
    main()
