# Registro de validação

Os notebooks 00–03 foram executados integralmente no Google Colab, em GPU T4:
**60 células de código concluídas, nenhum erro**. A execução técnica passou,
mas a qualidade de detecção na fazenda Derval foi baixa. O projeto preserva
esse resultado e investiga a escala dos objetos em uma etapa adicional.

O notebook 04 está em validação. Seus resultados ainda não estão disponíveis
nesta versão do registro; a hipótese e o protocolo estão em
[PLANO-RECORTES.md](PLANO-RECORTES.md).

## Ambiente e execução integral

Execução em 27/09/2026: Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128,
Ultralytics 8.3.203, NumPy 2.2.6 e Pillow 11.3.0. O ambiente completo está
no pacote de evidências. Treino inicial: 25 épocas, batch 8, entrada 640,
semente 42, 262,2 segundos de treinamento. O notebook 03 usa exatamente
os mesmos pesos ajustados do notebook 02.

| Notebook | Células executadas | Erros | Tempo total (s) |
|---|---:|---:|---:|
| 00 — Dados e problema | 16 | 0 | 15,4 |
| 01 — Baseline e contagem | 13 | 0 | 35,8 |
| 02 — Fine-tuning | 14 | 0 | 299,5 |
| 03 — Objetos pequenos | 17 | 0 | 69,8 |

A validação usou checkout limpo do GitHub, download raw e execução por
`nbconvert`. Fonte inicial: commit `0dc83dceaa35b0e3fddfb1d25415ffb8b4b9be82`.
O notebook 00 foi executado novamente após a correção do título da figura
anotada, no commit `6b50c14719741721c4a812c98604ad127db722df`.
Os hashes de código, dados, pesos, versões e relatórios estão preservados.

Para reproduzir a primeira sequência, em um diretório novo de `projeto-4`:

```bash
python scripts/validar_notebooks.py
```

O script executa 00–03, exige todas as células sem erros, extrai figuras e
empacota os relatórios e o checkpoint. Não sobrescreve um treinamento existente.
O notebook 04 é uma execução adicional, com protocolo próprio.

## Resultado inicial em Derval

São 60 imagens, 20 positivas, 111 caixas e dois voos. Cada confiança foi
escolhida apenas nas 60 imagens de validação. Detecção usa associação uma a uma
com IoU mínimo de 0,5; **F1 nesta tabela não é mAP**.

| Método | Confiança | TP / FP / FN | Precisão | Recall | F1 | MAE de contagem | Viés |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO, imagem inteira | 0,80 | 0 / 0 / 111 | 0 | 0 | 0 | 1,850 | −1,850 |
| Ajustado, imagem inteira | 0,20 | 6 / 24 / 105 | 0,200 | 0,054 | 0,085 | 1,683 | −1,350 |
| Mesmo ajustado, recortes | 0,30 | 5 / 302 / 106 | 0,016 | 0,045 | 0,024 | 4,667 | 3,267 |

Na validação, o F1 do ajustado foi 0,413 com imagem inteira e 0,371 com
recortes. O problema de generalização persistiu em Derval. Recortar somente
na inferência aumentou os falsos positivos em texturas do pasto.

Contar zero em todas as fotos já acerta exatamente 66,7% das imagens de
Derval, pois 40 estão vazias. Por isso o material também reporta métricas
nas imagens positivas: o MAE foi 5,55; 4,70; e 3,90, respectivamente.
Nenhum desses resultados sustenta uso operacional como contador de rebanho.

## Evidências e verificações

- [Pacote compacto de relatórios iniciais](assets/referencia-inicial.zip) e
  [índice com hashes](assets/referencia-inicial.json): predições, métricas,
  configuração, histórico do treino e relatórios da execução.
- [Pesos iniciais ajustados](assets/pesos-gado.zip) e
  [proveniência](assets/pesos-gado.json).
- [Auditoria das 300 imagens](assets/validacao-dados.json) e
  [auditoria das 62 novas capturas](assets/reserva-validacao.json).
- 27 testes originais aprovados: IoU, associação, NMS, coordenadas, contagem,
  parser de anotações e calibração restrita à validação.
- Oito testes adicionais aprovados para o preparo dos recortes: projeção,
  caixas parciais, amostragem determinística, integridade e isolamento do teste.
- 1.869 caixas conferidas contra a conversão oficial da Ultralytics; diferença
  máxima de 0,001024 pixel por arredondamento. Conversão RGB/BGR e inversão de
  letterbox verificadas. As caixas YOLO de Derval concordam com o XML VOC.
- Preparo real de 572 recortes de treino e 706 de validação, sem ler o teste;
  1.278 imagens e anotações conferidas por hash.

## Limites

Derval passou a ser diagnóstico exploratório para a nova receita, pois seus
resultados já foram observados. A avaliação adicional usa 62 capturas de cinco
voos ainda não usados, da pasta agregada `Other_farms`. Seus metadados não
comprovam independência geográfica, e todas essas fotos são positivas: não
estimam a taxa de falso alarme em imagens vazias.

Fotos próximas de um mesmo voo são correlacionadas. Uma única semente e
execução não estimam a variabilidade entre treinamentos. Fixar versões, dados
e configuração torna o procedimento repetível, sem prometer pesos idênticos
bit a bit entre máquinas. Contagens por foto não equivalem a um censo de
animais únicos entre fotos sobrepostas.
