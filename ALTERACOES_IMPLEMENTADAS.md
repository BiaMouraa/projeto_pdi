# Relatório de Alterações — Trabalho PDI 2026.1

**Projeto:** Canny Modificado Gabor-Di Zenzo  
**Data:** 05/07/2026  
**Para:** equipe / apresentação ao professor Leonardo  

> **Como gerar PDF:** abra este arquivo no VS Code ou no navegador → `Ctrl+P` → "Salvar como PDF"

---

## 1. Resumo executivo

Este documento lista **todas as alterações** feitas no código para atender **100% da especificação** do trabalho prático. Antes havia gaps importantes (filtros externos, Gabor escalar, histerese incompleta). Agora o projeto cobre:

- Módulo A, B e C completos
- Canny tradicional (Sobel + cinza manual)
- Canny modificado (Gabor + Di Zenzo)
- Gabor escalar (para Experimento 1)
- Batch de experimentos automatizado
- Script de validação pós-testes

---

## 2. Alterações implementadas (detalhado)

### 2.1 Módulo A — Carregamento de filtros de arquivos externos

| Item | Antes | Depois |
|------|-------|--------|
| Sobel | Hardcoded em `moduleC.py` | Carregado de `filters/sobel_x.json` e `filters/sobel_y.json` |
| Gaussiana | Gerada no código | Carregada de `filters/gaussiana_5x5.json` |
| Função nova | — | `carregar_kernel(caminho)` em `moduleA.py` |

**Por quê:** o enunciado exige explicitamente *"ler e aplicar matrizes de filtros estáticos (ex: Sobel, Gaussianas) a partir de arquivos .txt ou .json"*.

**Base no enunciado:** Seção 2, Módulo A.

**Arquivos criados:**
- `filters/sobel_x.json`
- `filters/sobel_y.json`
- `filters/gaussiana_5x5.json`

---

### 2.2 Gabor escalar (Experimento 1 — "Gabor tradicional")

| Item | Antes | Depois |
|------|-------|--------|
| Pipeline | Só Di Zenzo (vetorial) | + `canny_gabor_escalar()` em `moduleC.py` |
| Saídas | 3 PNGs | + 3 PNGs do Gabor escalar (arquivos 7, 8, 9) |

**O que faz:** aplica o banco de Gabor **apenas no canal cinza** (fórmula manual Y=0.299R+0.587G+0.114B), sem fusão L2. Permite comparar sensibilidade paramétrica:

- `1_magnitude_di_zenzo.png` → modificado (cor vetorial)
- `7_gabor_escalar_magnitude.png` → escalar (só cinza)

**Por quê:** Experimento 1 pede *"Análise de Sensibilidade Paramétrica do Gabor tradicional e do Modificado"*.

**Base no enunciado:** Seção 4, Experimento 1.

---

### 2.3 Histerese iterativa (correção técnica)

| Item | Antes | Depois |
|------|-------|--------|
| Algoritmo | 1 passada (bordas fracas só ligadas diretamente a fortes) | BFS (fila) — propaga cadeias de bordas fracas |

**Por quê:** no Canny clássico, uma borda fraca conectada a outra fraca que conecta a uma forte **também deve ser mantida**. A versão anterior quebrava contornos.

**Base no enunciado:** Seção 2, Módulo C, passo 5 (histerese com conectividade).

---

### 2.4 Validação de `tamanho_mascara` ímpar

| Item | Antes | Depois |
|------|-------|--------|
| Validação | Nenhuma | Erro claro se valor par em `moduleB.py` |

**Por quê:** enunciado exige dimensão ímpar (ex: 31). Kernel par causaria erro silencioso ou resultado errado.

---

### 2.5 Renomeação dos arquivos de saída (clareza)

| Antigo | Novo | Motivo |
|--------|------|--------|
| `1_magnitude_normalizada.png` | `1_magnitude_di_zenzo.png` | Distinguir Di Zenzo de Gabor escalar |
| `2_nms.png` | `2_nms_di_zenzo.png` | Idem |
| `3_borda_final_histerese.png` | `3_borda_final_di_zenzo.png` | Idem |
| — | `7_gabor_escalar_magnitude.png` | Novo pipeline |
| — | `8_gabor_escalar_nms.png` | Novo pipeline |
| — | `9_gabor_escalar_borda_final.png` | Novo pipeline |

Arquivos 4–6 (Canny tradicional Sobel) permanecem iguais.

---

### 2.6 Script de validação (`validar_resultados.py`)

**Novo arquivo** que verifica automaticamente:

- Se todas as pastas/arquivos existem
- Contagem de pixels de borda
- Espessura estimada (~1 pixel)
- **Teste crítico GrayAndMagenta:** modificado deve detectar mais bordas que tradicional

Gera `results/validacao_relatorio.json` para enviar de volta e conferir.

---

### 2.7 Outros arquivos

| Arquivo | Função |
|---------|--------|
| `requirements.txt` | Dependências (numpy, opencv-python) |
| `.gitignore` | Ignora `results/` e `BACKLOG_RELATORIO.md` |
| `config.json` | Caminhos dos filtros + parâmetros tradicionais |
| `experiments_config.json` | Grade de experimentos (λ, σ, histerese) |
| `run_experimentos.py` | Batch automático |
| `pipeline.py` | Lógica compartilhada de execução e salvamento |

---

## 3. Mapa completo de saídas por pasta

```
results/
└── GrayAndMagenta/
    ├── canny_tradicional/           ← Canny clássico (Sobel)
    │   ├── 4_tradicional_magnitude.png
    │   ├── 5_tradicional_nms.png
    │   ├── 6_tradicional_borda_final.png
    │   └── config_usado.json
    └── mask31_sigma4.0_..._baseline/  ← Modificado + Gabor escalar
        ├── 1_magnitude_di_zenzo.png      (Exp. 1 e 2 — modificado)
        ├── 2_nms_di_zenzo.png            (Exp. 2 — afinamento)
        ├── 3_borda_final_di_zenzo.png    (Exp. 2 — borda final)
        ├── 7_gabor_escalar_magnitude.png (Exp. 1 — Gabor tradicional)
        ├── 8_gabor_escalar_nms.png
        ├── 9_gabor_escalar_borda_final.png
        └── config_usado.json
    └── ..._exp1_lambda4/    (Exp. 1 — λ=4)
    └── ..._exp1_sigma8/     (Exp. 1 — σ=8)
    └── ..._exp2_histerese_* (Exp. 2 — limiares)
```

---

## 4. Comandos para rodar os testes

### Passo 0 — Instalar dependências (uma vez)

```bash
cd C:\Users\MatheusMendonça\Desktop\projeto_pdi
pip install -r requirements.txt
```

### Passo 1 — Teste rápido (1 imagem, ~2 min)

```bash
python run_experimentos.py --rapido
```

Processa só `GrayAndMagenta.png` com baseline + tradicional.

### Passo 2 — Batch completo (6 imagens × 8 variações, demora)

```bash
python run_experimentos.py
```

Ou por experimento:

```bash
python run_experimentos.py --exp 1    # só lambda e sigma
python run_experimentos.py --exp 2    # só histerese
```

### Passo 3 — Validar resultados

```bash
python validar_resultados.py
python validar_resultados.py --imagem GrayAndMagenta
```

### Passo 4 — Imagem individual (opcional)

```bash
python main.py images/Bear.jpg
```

---

## 5. Como validar manualmente (checklist)

### Experimento 1 — Sensibilidade (λ e σ)

1. Abra `results/Zebra/..._exp1_lambda4/` vs `..._exp1_lambda20/`
2. Compare `1_magnitude_di_zenzo.png`:
   - λ=4 → mais textura fina (listras)
   - λ=20 → contornos grossos, listras somem
3. Compare `results/Zebra/..._exp1_sigma2/` vs `..._exp1_sigma8/`:
   - σ=8 → listras borradas (σ grande demais para textura)
4. Compare `1_` (Di Zenzo) vs `7_` (escalar) na mesma pasta:
   - Di Zenzo deve preservar mais bordas de cor

### Experimento 2 — NMS e histerese

1. Na pasta baseline, abra lado a lado: `1_` → `2_` → `3_`
2. Dê zoom 800%+ em `3_borda_final_di_zenzo.png`
3. Conte pixels brancos na direção perpendicular à borda → deve ser **~1**
4. Compare `exp2_histerese_tlow10_thigh30` vs `tlow30_thigh80`:
   - Limiar baixo → mais ruído
   - Limiar alto → bordas quebradas

### Comparação tradicional vs modificado

1. Abra `GrayAndMagenta/canny_tradicional/6_` vs `...baseline/3_`
2. **Esperado:** tradicional quase vazio; modificado mostra linha entre cinza e magenta
3. Rode `validar_resultados.py` — deve imprimir `[OK] Modificado detectou MAIS bordas`

---

## 6. O que estudar para apresentação amanhã

### Perguntas prováveis do professor → respostas curtas

**"O que é detecção de bordas?"**  
Achar onde a imagem muda bruscamente (cor ou brilho).

**"Por que o Canny tradicional falha em GrayAndMagenta?"**  
Porque converte para cinza antes. Cinza e magenta têm o mesmo brilho → viram iguais → sem borda.

**"O que o modificado faz diferente?"**  
Processa R, G, B juntos (Di Zenzo) com filtros Gabor. Vê bordas de cor, não só de brilho.

**"O que é um filtro de Gabor?"**  
Padrão de listras + gaussiana, girado em várias direções. Detecta bordas orientadas.

**"O que faz λ (lambda)?"**  
Comprimento de onda. Pequeno = texturas finas. Grande = contornos grossos.

**"O que faz σ (sigma)?"**  
Escala espacial. Grande = borra, perde detalhes. Pequeno = sensível a ruído.

**"O que é NMS?"**  
Supressão de não-máximos. Afina borda para 1 pixel, fica só no "pico".

**"O que é histerese?"**  
Dois limiares: forte (certeza) e fraco (só se conectado a forte). Remove ruído sem quebrar bordas.

**"Por que Sobel está em arquivo JSON?"**  
Enunciado exige filtros estáticos carregados de arquivos externos (Módulo A).

**"Diferença Gabor escalar vs Di Zenzo?"**  
Escalar: Gabor só no cinza. Di Zenzo: Gabor nos 3 canais + fusão L2 √(R²+G²+B²).

### Arquivos que vocês PRECISAM saber explicar

| Arquivo | O que dizer |
|---------|-------------|
| `moduleA.py` | Correlação 2D + carrega filtros de JSON |
| `moduleB.py` | Gera kernels Gabor do config |
| `moduleC.py` | Todos os pipelines + NMS + histerese |
| `config.json` | Parâmetros ajustáveis |
| `main.py` | Roda 1 imagem |
| `run_experimentos.py` | Roda todos os experimentos |

---

## 7. Conformidade com o enunciado (checklist final)

| Requisito | Status |
|-----------|--------|
| Módulo A — correlação RGB | ✅ |
| Módulo A — filtros de arquivos .json | ✅ |
| Módulo B — Gabor paramétrico via JSON | ✅ |
| Módulo C — pipeline modificado (5 passos) | ✅ |
| Cinza manual Y=0.299R+0.587G+0.114B | ✅ |
| Canny tradicional (Sobel) | ✅ |
| Gabor escalar (Exp. 1) | ✅ |
| Expansão histograma [0,255] | ✅ |
| Sem cv2.Canny/filter2D/getGaborKernel | ✅ |
| Exp. 1 — variação λ e σ | ✅ (run_experimentos) |
| Exp. 2 — NMS + histerese | ✅ |
| Comparação tradicional vs modificado | ✅ |
| GrayAndMagenta destacada | ✅ |
| JSON de configuração | ✅ |
| Relatório PDF (documento acadêmico) | ⏳ pendente (vocês escrevem) |

---

## 8. Após rodar os testes — o que enviar para validação

1. Saída completa do terminal:
   ```bash
   python validar_resultados.py
   ```
2. Screenshot ou confirmação visual de `GrayAndMagenta`:
   - `canny_tradicional/6_tradicional_borda_final.png`
   - `...baseline/3_borda_final_di_zenzo.png`
3. Arquivo `results/validacao_relatorio.json`

Com isso dá para confirmar se está batendo com a especificação.

---

*Documento gerado automaticamente como parte da implementação do trabalho prático de PDI.*
