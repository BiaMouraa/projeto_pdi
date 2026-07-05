# Backlog — Template de Relatório (PDI 2026.1)

> Arquivo de referência para montar o relatório PDF após conclusão da implementação.
> Não entregar este arquivo — é nota interna do projeto.

## Contexto do trabalho

- **Disciplina:** Introdução ao Processamento Digital de Imagens
- **Período:** 2026.1 | **Professor:** Leonardo
- **Entrega:** 01/07/2026 (PDF + código-fonte + configs JSON)
- **Apresentação:** relatório impresso + todos os integrantes presentes

## Objetivo central

Implementar, analisar e comparar:
1. **Canny tradicional** (escalar, tons de cinza)
2. **Canny modificado Gabor-Di Zenzo** (vetorial, RGB)

## Problema que o trabalho resolve

Canny tradicional converte RGB → cinza antes dos gradientes, destruindo **bordas cromáticas** (mesmo brilho, cores diferentes). O modificado preserva cor via fusão L2 (Di Zenzo) no domínio das derivadas.

**Imagem-chave para discussão:** `GrayAndMagenta.png` — cinza e magenta com brilho similar; tradicional falha, modificado deve detectar.

## Imagens de teste obrigatórias

- Bear.jpg
- FCBarcelona.png
- GrayAndMagenta.png
- PlacaMercosul.webp
- VintageCar.png
- Zebra.png

## Pipeline modificado (Gabor-Di Zenzo)

1. Filtragem Gabor por orientação (R, G, B separados — Módulo A)
2. Fusão L2: `Magn = sqrt(R² + G² + B²)`
3. Redução por máximo entre orientações
4. NMS (afinamento 1 pixel)
5. Histerese (t_low, t_high)

## Pipeline tradicional (Canny clássico)

1. Cinza manual: `Y = 0.299R + 0.587G + 0.114B`
2. Suavização Gaussiana
3. Gradientes Sobel (Gx, Gy)
4. Magnitude + orientação
5. NMS + Histerese (mesmas funções, limiares próprios)

## Arquivos PNG gerados

| Arquivo | Significado | Uso no relatório |
|---------|-------------|------------------|
| `1_magnitude_normalizada.png` | Mapa Di Zenzo (bordas grossas) | Exp. 1 sensibilidade, Exp. 2 antes do NMS |
| `2_nms.png` | Após supressão de não-máximos | Exp. 2 afinamento |
| `3_borda_final_histerese.png` | Borda binária final | Exp. 2 zoom 1px, comparação |
| `4_tradicional_magnitude.png` | Magnitude Sobel | Comparação tradicional |
| `5_tradicional_nms.png` | NMS tradicional | Comparação |
| `6_tradicional_borda_final.png` | Borda Canny clássico | Comparação (destaque GrayAndMagenta) |

## Experimento 1 — Sensibilidade paramétrica

**Para cada imagem**, variar parâmetros Gabor e documentar impacto no mapa de magnitudes:

| Parâmetro | Valores sugeridos | O que observar |
|-----------|-------------------|----------------|
| `lambda_` (λ) | 4, 10, 20 | λ↓ texturas finas; λ↑ macrocontornos |
| `sigma` (σ) | 2, 4, 8 | σ grande em Zebra → borra listras |

**Perguntas obrigatórias:**
- λ isola alta vs baixa frequência?
- σ excessivo em imagem texturizada — o que acontece?

## Experimento 2 — NMS e histerese

- Lado a lado: magnitude | NMS | borda final
- Zoom digital provando **1 pixel de largura**
- Justificar escolha de `t_low` e `t_high`

## Comparação tradicional vs modificado

- Especial detalhe em **GrayAndMagenta.png**
- Tabela ou figura lado a lado por imagem

## Seções obrigatórias do relatório PDF

1. **Introdução** — contextualização, fundamentação teórica, objetivos
2. **Materiais e métodos** — atividades, ferramentas, conhecimentos
3. **Resultados** — figuras dos experimentos
4. **Discussão** — dificuldades, comentários críticos
5. **Conclusão**

## Restrições de implementação

- Proibido: `cv2.Canny`, `cv2.filter2D`, `cv2.getGaborKernel`
- Permitido: NumPy, cv2/Pillow só para ler/salvar imagens

## Parâmetros default (config.json)

```json
{
  "gabor": {
    "tamanho_mascara": 31,
    "sigma": 4.0,
    "lambda_": 10.0,
    "gamma": 0.5,
    "psi": 0.0,
    "orientacoes_graus": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
  },
  "histerese": { "t_low": 20, "t_high": 50 },
  "canny_tradicional": {
    "sigma_gauss": 1.4,
    "tamanho_mascara_gauss": 5,
    "t_low": 50,
    "t_high": 150
  }
}
```

## Estrutura de resultados no disco

```
results/
└── Bear/
    ├── canny_tradicional/
    │   └── 4/5/6_tradicional_*.png + config_usado.json
    └── mask31_sigma4.0_lambda10.0_.../
        ├── 1/2/3_*.png
        └── config_usado.json
```

## Trilha de estudos (referência para apresentação)

1. Pixel, RGB, escala de cinza manual
2. Borda, gradiente, filtros/kernels
3. Canny tradicional (Gauss + Sobel + NMS + histerese)
4. Filtros de Gabor (σ, λ, orientações)
5. Di Zenzo (fusão vetorial L2)
6. Experimentos e interpretação visual

## Pendências pós-implementação (para o relatório)

- [ ] Rodar `run_experimentos.py` e revisar todas as pastas
- [ ] Selecionar melhores figuras para cada seção
- [ ] Screenshots de zoom (1 pixel) para Exp. 2
- [ ] Redigir justificativa dos limiares
- [ ] Discussão GrayAndMagenta (tradicional vs modificado)
- [ ] Montar template PDF com figuras numeradas
