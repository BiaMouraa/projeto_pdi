# Issue 4 - Avaliacao das metricas de busca semantica

- Data da execucao: 2026-08-04 22:46:36
- Imagem de teste: `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0422.jpg`
- Classe de referencia para coerencia: `iris`
- Repeticoes de benchmark por consulta: `5`

## Consultas SQL implementadas

- Distancia Euclidiana (`<->`):
```sql
SELECT image_id, class_name, file_path, embedding <-> $1::vector AS score
FROM flower_embeddings
ORDER BY embedding <-> $1::vector
LIMIT $2;
```

- Distancia de Cosseno (`<=>`):
```sql
SELECT image_id, class_name, file_path, embedding <=> $1::vector AS score
FROM flower_embeddings
ORDER BY embedding <=> $1::vector
LIMIT $2;
```

## Resultado comparativo

### Metrica `euclidiana` (operador `<->`)
- Top-5: tempo medio `2.43 ms` (min `1.53` / max `5.52`)
- Top-5: coerencia visual `100.00%`
- Top-5: primeiros resultados
  - 01. `image_0469.jpg` | classe `iris` | score `0.000035` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0469.jpg`
  - 02. `image_0422.jpg` | classe `iris` | score `0.000035` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0422.jpg`
  - 03. `image_0470.jpg` | classe `iris` | score `11.539225` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0470.jpg`
  - 04. `image_0433.jpg` | classe `iris` | score `11.550586` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0433.jpg`
  - 05. `image_0425.jpg` | classe `iris` | score `12.552876` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0425.jpg`
- Top-10: tempo medio `1.58 ms` (min `1.51` / max `1.77`)
- Top-10: coerencia visual `100.00%`
- Top-10: primeiros resultados
  - 01. `image_0469.jpg` | classe `iris` | score `0.000035` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0469.jpg`
  - 02. `image_0422.jpg` | classe `iris` | score `0.000035` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0422.jpg`
  - 03. `image_0470.jpg` | classe `iris` | score `11.539225` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0470.jpg`
  - 04. `image_0433.jpg` | classe `iris` | score `11.550586` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0433.jpg`
  - 05. `image_0425.jpg` | classe `iris` | score `12.552876` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0425.jpg`
  - 06. `image_0416.jpg` | classe `iris` | score `12.708742` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0416.jpg`
  - 07. `image_0443.jpg` | classe `iris` | score `13.068193` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0443.jpg`
  - 08. `image_0424.jpg` | classe `iris` | score `13.068193` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0424.jpg`
  - 09. `image_0406.jpg` | classe `iris` | score `13.274171` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0406.jpg`
  - 10. `image_0475.jpg` | classe `iris` | score `13.632705` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0475.jpg`

### Metrica `cosseno` (operador `<=>`)
- Top-5: tempo medio `1.46 ms` (min `1.35` / max `1.79`)
- Top-5: coerencia visual `100.00%`
- Top-5: primeiros resultados
  - 01. `image_0469.jpg` | classe `iris` | score `0.000000` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0469.jpg`
  - 02. `image_0422.jpg` | classe `iris` | score `0.000000` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0422.jpg`
  - 03. `image_0470.jpg` | classe `iris` | score `0.101098` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0470.jpg`
  - 04. `image_0433.jpg` | classe `iris` | score `0.101361` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0433.jpg`
  - 05. `image_0425.jpg` | classe `iris` | score `0.119206` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0425.jpg`
- Top-10: tempo medio `1.41 ms` (min `1.34` / max `1.47`)
- Top-10: coerencia visual `100.00%`
- Top-10: primeiros resultados
  - 01. `image_0469.jpg` | classe `iris` | score `0.000000` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0469.jpg`
  - 02. `image_0422.jpg` | classe `iris` | score `0.000000` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0422.jpg`
  - 03. `image_0470.jpg` | classe `iris` | score `0.101098` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0470.jpg`
  - 04. `image_0433.jpg` | classe `iris` | score `0.101361` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0433.jpg`
  - 05. `image_0425.jpg` | classe `iris` | score `0.119206` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0425.jpg`
  - 06. `image_0416.jpg` | classe `iris` | score `0.122712` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0416.jpg`
  - 07. `image_0475.jpg` | classe `iris` | score `0.134011` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0475.jpg`
  - 08. `image_0403.jpg` | classe `iris` | score `0.134096` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0403.jpg`
  - 09. `image_0406.jpg` | classe `iris` | score `0.138593` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0406.jpg`
  - 10. `image_0455.jpg` | classe `iris` | score `0.143197` | `/Users/matheusmendonca/Desktop/projeto_pdi/projeto_final/data/processed/iris/image_0455.jpg`

## Metrica definitiva recomendada

- Metrica escolhida: `cosseno`
- Justificativa: Empate em coerencia e menor latencia media.

## Observacao

- Se necessario, repita com outras imagens de teste para validar estabilidade da escolha.
