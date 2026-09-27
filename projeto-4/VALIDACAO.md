# Registro de validação

**Em preparação.** Este registro será preenchido com a execução integral dos
quatro notebooks antes de considerar a entrega pronta. Ainda não há métricas de
desempenho publicadas nesta versão do branch.

## Verificações locais concluídas

Os 27 testes do motor de contagem e do protocolo de avaliação passaram. Cobrem
IoU, associação de caixas uma a uma, NMS, coordenadas dos recortes, erros de
contagem, leitura de anotações e proibição de calibrar a confiança fora da
validação. São testes controlados; não medem acurácia em fotografias reais.

As anotações YOLO das 300 imagens selecionadas passaram pela leitura estrita de
classe, dimensões e limites normalizados. Os notebooks passaram pela validação
de estrutura e sintaxe, com até 12 linhas por célula de código.

## Execução integral

Na raiz de `projeto-4`, com os dados distribuídos e GPU disponível:

```bash
python scripts/validar_notebooks.py
```

O script executa os notebooks com `nbconvert`, exige todas as células concluídas
sem erros, registra versões e hardware, extrai figuras e empacota relatórios e o
checkpoint. O treino ocorre apenas no notebook 02, com 25 épocas, batch 8,
imagem de entrada 640 e semente 42. O notebook 03 usa o mesmo checkpoint.

A execução usa um diretório novo de treinamento. Uma repetição precisa de outro
nome de execução; não apague resultados anteriores para contornar essa proteção.

## Escopo das conclusões

As futuras métricas descreverão as 60 imagens da fazenda de teste deste recorte.
São apenas 20 imagens positivas e dois voos correlacionados. Não demonstram
desempenho universal em outras fazendas ou sensores. Resultados de detecção usam
IoU mínimo de 0,5 e um limiar de confiança escolhido na validação; não são mAP.

Tempos dependem da GPU e incluem inicialização na primeira inferência. Uma única
semente e execução não estimam a variabilidade entre treinamentos. Fixar versões,
dados e configuração torna o procedimento repetível, sem prometer igualdade de
pesos bit a bit entre máquinas.
