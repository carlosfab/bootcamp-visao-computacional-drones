# Experimento adicional: compatibilizar a escala de treinamento e inferência

Plano definido antes de treinar o modelo adicional e antes de executar modelos
na reserva externa. A implementação e este plano são versionados no branch.

## Motivação observada

O experimento inicial foi executado integralmente e será preservado. Na validação,
o modelo treinado em fotografias inteiras a 640 pixels obteve F1 de 0,413; aplicar
recortes somente na inferência produziu F1 de 0,371. Em Derval, os valores foram
0,085 e 0,024. A inspeção mostrou omissões e falsos positivos na textura do pasto.
A auditoria de arquivos e coordenadas não encontrou um defeito que explicasse
esses resultados.

A hipótese é que **treinar com recortes na mesma escala da inferência** forneça
mais detalhe útil e reduza a diferença de escala entre os dois momentos. Isso
não garante superar a mudança de aparência entre fazendas ou resolver oclusões.

Este é um experimento adicional concebido depois de observar Derval. Portanto,
a reavaliação nessa fazenda é exploratória. Seus resultados anteriores não serão
apagados nem apresentados como se esta receita tivesse sido escolhida antes deles.

## Receita definida

- Fonte de treino e validação: as mesmas 180 e 60 fotografias originais do recorte
  `gado-icaerus-v1`, com partições preservadas.
- Janelas de 640 pixels, sobreposição nominal de 20%, cobrindo cada fotografia.
- Coordenadas de rótulos projetadas e limitadas à janela. Rejeitar uma janela
  inteira quando algum bovino parcialmente visível tiver menos de 50% da caixa
  original dentro dela ou um lado resultante inferior a 2 pixels. Assim evitamos
  manter uma imagem com fragmentos de bovinos deliberadamente sem rótulo.
- Treino: todas as janelas positivas elegíveis e até o mesmo número de negativas,
  selecionadas por uma ordenação SHA-256 determinística. Validação: todas as
  janelas elegíveis, sem esse balanceamento. Um recorte herda a partição da foto.
- YOLO11n inicializado com pesos COCO; 20 épocas, batch 16, entrada 640, semente 42,
  dois workers, modo determinístico, sem cache. Mosaic desativado; ampliação e
  redução de escala limitadas pelo argumento `scale=0.25`.
- Selecionar `best.pt` pela validação durante o treinamento. Calibrar a confiança
  por F1 nas 60 fotografias originais de validação, processadas com recortes.
  Empates: menor MAE, depois maior confiança. NMS global com IoU 0,5.

O comparador adicional é o YOLO11n COCO com a mesma inferência em recortes, cujo
limiar também é escolhido separadamente nas mesmas imagens de validação.

## Avaliação externa reservada

As 62 fotografias da pasta de origem `Other_farms`, ainda fora dos experimentos
anteriores, serão avaliadas somente depois de congelar pesos e limiares. A seleção
inclui todas as imagens desses cinco voos; não escolhemos casos pelas previsões.
São 1.355 caixas e somente imagens positivas.

`Other_farms` é uma pasta agregada. Seus metadados não demonstram que ela represente
uma única fazenda nem que as propriedades sejam geograficamente independentes do
treino. A reserva permite avaliar novas capturas deste acervo. Não demonstra
generalização para qualquer fazenda, e não mede diretamente a taxa de falsos
positivos em fotografias sem bovinos anotados.

Reportar precisão, recall, F1 com IoU mínimo de 0,5, MAE, viés, contagem exata,
tempo e casos de erro. Esses resultados não são mAP. Não haverá escolha de outra
receita motivada por essa reserva sob a alegação de que ela continua intocada.
