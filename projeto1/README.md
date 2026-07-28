# Pipeline de Canny Modificado com Gabor-Di Zenzo

Detecção de bordas coloridas com filtros de Gabor + fusão vetorial Di Zenzo, comparado ao Canny tradicional (Sobel).

## Estrutura

| Arquivo/Pasta | Função |
|---------------|--------|
| `moduleA.py` | Correlação 2D RGB + `carregar_kernel()` de `.json`/`.txt` |
| `moduleB.py` | Banco de filtros Gabor paramétrico |
| `moduleC.py` | Pipelines (Di Zenzo, Gabor escalar, Canny tradicional, NMS, histerese) |
| `pipeline.py` | Execução e salvamento compartilhado |
| `main.py` | Processa uma imagem |
| `run_experimentos.py` | Batch de experimentos para o relatório |
| `validar_resultados.py` | Validação automática pós-testes |
| `config.json` | Parâmetros Gabor, histerese, caminhos dos filtros |
| `experiments_config.json` | Grade de experimentos (λ, σ, histerese) |
| `filters/` | Kernels Sobel e Gaussiana em JSON |
| `ALTERACOES_IMPLEMENTADAS.md` | Relatório de alterações para a equipe |

## Instalação

```bash
pip install -r requirements.txt
```

## Comandos

```bash
# Teste rápido (GrayAndMagenta)
python run_experimentos.py --rapido

# Batch completo (6 imagens × 8 variações)
python run_experimentos.py

# Validar resultados
python validar_resultados.py

# Uma imagem
python main.py images/GrayAndMagenta.png
```

## Saídas

**Modificado + Gabor escalar** (`results/<imagem>/<params>/`):

| Arquivo | Descrição |
|---------|-----------|
| `1_magnitude_di_zenzo.png` | Mapa Di Zenzo (modificado) |
| `2_nms_di_zenzo.png` | NMS modificado |
| `3_borda_final_di_zenzo.png` | Borda final modificado |
| `7_gabor_escalar_magnitude.png` | Gabor escalar (Exp. 1) |
| `8_gabor_escalar_nms.png` | NMS escalar |
| `9_gabor_escalar_borda_final.png` | Borda final escalar |

**Canny tradicional** (`results/<imagem>/canny_tradicional/`):

| Arquivo | Descrição |
|---------|-----------|
| `4_tradicional_magnitude.png` | Magnitude Sobel |
| `5_tradicional_nms.png` | NMS |
| `6_tradicional_borda_final.png` | Borda final |

Cada pasta inclui `config_usado.json`.
