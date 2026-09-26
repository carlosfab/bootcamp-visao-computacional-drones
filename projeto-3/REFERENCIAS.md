# Referências e atualização técnica

Consulta: **26 de setembro de 2026 (UTC)**. Foram examinados o índice recente do blog Roboflow, artigos relacionados a anotação, coordenadas e estado temporal, os cookbooks oficiais e a documentação das APIs. Os exemplos executáveis usam bibliotecas Python locais; não exigem Workflows, Inference hospedado, conta Roboflow ou chave de API.

As páginas `latest` e o blog podem mudar após a consulta. Por isso, a reprodução depende das versões fixadas e dos arquivos verificados, não apenas da data de um tutorial. O relatório de execução acompanha o material; documentação consultada e código executado são evidências diferentes.

## Fundamentação

| Fonte primária | Uso no material |
|:--|:--|
| Lowe. [Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), IJCV, 2004. | Distinguir localização de pontos de interesse e construção de descritores locais; invariância e correspondência têm hipóteses e limites. |
| OpenCV. [ORB: Oriented FAST and Rotated BRIEF](https://docs.opencv.org/4.13.0/d1/d89/tutorial_py_orb.html). | Exemplo de combinação entre detector e descritor; uma característica local não equivale a um objeto ou ID temporal. |
| Bewley et al. [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), 2016. | Rastreamento por detecção, previsão de movimento e associação entre quadros. |
| Wojke, Bewley e Paulus. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402), 2017. | Contextualizar aparência aprendida e associação; esse componente não faz parte do ByteTrack usado nas aulas. |
| Zhang et al. [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), versão de 2022. | Recuperação de trajetórias com detecções de menor confiança. Os resultados publicados pertencem aos benchmarks do artigo, não aos vídeos desta aula. |

## Documentação de implementação

| Referência oficial | Itens conferidos |
|:--|:--|
| [Supervision: `Detections`](https://supervision.roboflow.com/latest/detection/core/) | Campos, alinhamento por instância, filtros booleanos, `from_inference`, `from_ultralytics` e coordenadas de âncoras. |
| [Supervision: conversores](https://supervision.roboflow.com/latest/detection/utils/converters/) | `xyxy_to_xywh`, `xywh_to_xyxy` e `xcycwh_to_xyxy`; `xywh` significa origem superior esquerda mais dimensões. |
| [Supervision: filtros](https://supervision.roboflow.com/latest/how_to/filter_detections/) | Seleção por classe, confiança, geometria e condições combinadas. |
| [Supervision: anotadores](https://supervision.roboflow.com/latest/detection/annotators/) | Composição de caixas, rótulos, pontos, cantos e rastros; estratégia de cor por classe ou identidade. |
| [Supervision: vídeo](https://supervision.roboflow.com/latest/utils/video/) | Leitura por gerador, metadados de vídeo e gravação com `VideoSink`. |
| [Supervision: migração de tracking](https://supervision.roboflow.com/latest/trackers/) | `sv.ByteTrack` depreciado desde 0.28.0; substituição por `trackers.ByteTrackTracker`, com `update()` no lugar de `update_with_detections()`. |
| [Trackers: ByteTrack](https://trackers.roboflow.com/latest/trackers/bytetrack/) | Associação em duas etapas, parâmetros, ciclo de vida, buffer temporal e significado de `tracker_id=-1`. |
| [RF-DETR: inferência](https://rfdetr.roboflow.com/latest/learn/run/detection/) | RF-DETR Medium, entrada RGB, predição e retorno nativo de `sv.Detections`. |
| [RF-DETR: migração](https://rfdetr.roboflow.com/latest/getting-started/migration/) | Conferência da interface atual em relação às receitas anteriores. |

Também foi inspecionado o código distribuído nos wheels oficiais de `supervision==0.30.5`, `trackers==2.6.1` e `rfdetr==1.11.0`, além da documentação. Isso evita transferir silenciosamente defaults de implementações antigas para a aula atual.

## Blog Roboflow: seleção recente e aplicação didática

A seleção abaixo prioriza relação com esta aula. Artigos sobre treinamento de outros modelos, agentes ou implantação servem como contexto e não acrescentam dependências aos notebooks.

| Publicação | Artigo | Aplicação e limite |
|:--|:--|:--|
| 22/09/2026 | [Build Agentic Computer Vision with Roboflow Workflows](https://blog.roboflow.com/agentic-computer-vision/) | Artigo recente consultado nas seções de percepção e memória temporal. Distingue previsão por quadro de estado e eventos. O produto Workflows e a camada de agentes estão fora desta prática Python. |
| 28/08/2026 | [How to Build a Parking Lot Monitoring System with Computer Vision](https://blog.roboflow.com/build-a-parking-lot-monitoring-system/) | Exemplo recente para distinguir ocupação instantânea e fluxo temporal. Sua aplicação é montada em Workflows; não é uma receita equivalente de `supervision` local. |
| 27/08/2026 | [Object Detection API: Best Hosted Options in 2026](https://blog.roboflow.com/object-detection-api/) | Conferência do esquema de caixas centradas no JSON e do adaptador `from_inference`. No notebook, a conversão é demonstrada localmente com uma previsão real, sem chamada hospedada. |
| 14/08/2026 | [mAP@0.5 vs. mAP@0.5:0.95](https://blog.roboflow.com/map-0-5-vs-map-0-5-0-95/) | Reforça a necessidade de anotações de referência para avaliação quantitativa. Vídeo anotado e score de confiança não substituem métricas de validação. |
| 30/06/2026 | [Build a Drone-Based Security System with Computer Vision](https://blog.roboflow.com/drone-based-security-reconnaissance-system/) | A seção de âncoras motiva escolher o ponto geométrico conforme a perspectiva. O centro inferior não representa automaticamente contato com o solo em qualquer cena aérea. |
| 26/06/2026 | [How to Fine-Tune RF-DETR Keypoints on Custom Data](https://blog.roboflow.com/train-rf-detr-keypoint/) | Contextualiza pontos semânticos aprendidos e sua representação. Detecção de keypoints semânticos é distinta de SIFT/ORB e de IDs de tracking. |
| 15/04/2026 | [How to Use Roboflow to Create Conditional Annotators](https://blog.roboflow.com/create-conditional-annotators/) | Inspira seleção espacial e estilo de anotação por condição. O artigo integra Workflows e ainda contém `sv.ByteTrack`; a aula usa a API atual do pacote `trackers`. |

## Tutoriais e vídeo de entrada

O [cookbook oficial de contagem por linha](https://github.com/roboflow/supervision/blob/f132aa01da7d42b36f8615f923285e1839b5e05e/docs/notebooks/count-objects-crossing-the-line.ipynb), congelado na revisão `f132aa01da7d42b36f8615f923285e1839b5e05e`, fornece a referência prática RF-DETR Medium → ByteTrackTracker → anotadores. O notebook `02_tracking.ipynb` isola o rastreamento, simplifica as células e não implementa a contagem desse cookbook.

O [tutorial em vídeo de 20/01/2023](https://www.youtube.com/watch?v=OS5qI9YBkfk) e o [post de tracking e contagem com título YOLOv8](https://blog.roboflow.com/yolov8-tracking-and-counting/) são referências históricas da sequência de ensino. O código desses materiais evoluiu em momentos diferentes. Eles não determinam os nomes das classes, métodos ou defaults das versões atuais.

**Vídeo dos notebooks 01 e 02:** [vehicles-1280x720.mp4, hospedado pela Roboflow](https://storage.googleapis.com/com-roboflow-marketing/supervision/cookbooks/vehicles-1280x720.mp4), também referenciado pelo cookbook. É um vídeo de entrada sem as anotações produzidas nesta aula.

- Tamanho do arquivo verificado: `2839625` bytes.
- SHA-256: `9202baceb8b07949a06a9df821fb11ec536d35cef41eff75beb20701ff2d7f22`.
- Notebook 01: inspeção do quadro de índice 30.
- Notebook 02: primeiros oito segundos, todos os quadros, mantendo o FPS original.
- O download é feito da fonte pública original; o material não depende de publicação do branch local.

Os ativos da rotatória e suas fontes específicas estão descritos no notebook do projeto final. O uso educacional de um exemplo público não transforma esse vídeo em dataset anotado nem atribui uma licença nova ao conteúdo original.

## Versões e decisões de compatibilidade

Os metadados oficiais do PyPI foram consultados diretamente. As versões disponíveis dos três componentes centrais eram [Supervision 0.30.5](https://pypi.org/project/supervision/0.30.5/), [Trackers 2.6.1](https://pypi.org/project/trackers/2.6.1/) e [RF-DETR 1.11.0](https://pypi.org/project/rfdetr/1.11.0/). O material fixa essas versões, com o conjunto completo em `requirements.txt`.

| Aspecto | Decisão para a aula |
|:--|:--|
| Bibliotecas centrais | `supervision==0.30.5`, `trackers==2.6.1`, `rfdetr==1.11.0`. |
| Execução de redes | `torch==2.8.0` e `torchvision==0.23.0`; escolha de compatibilidade, sem afirmar que sejam as versões mais recentes. |
| Modelo YOLO do projeto final | `ultralytics==8.4.163`, preservando o checkpoint especializado selecionado. Uma atualização de biblioteca não troca os pesos do modelo. |
| Matrizes e vídeo | `numpy==2.3.5`, `opencv-python==5.0.0.93`, `imageio-ffmpeg==0.6.0`. |
| Cor de imagens | Arrays OpenCV em BGR; RF-DETR recebe RGB por `cv2.cvtColor`. |
| Convenção espacial | Origem superior esquerda; caixas `xyxy` em pixels da imagem fornecida; índices de arrays na ordem `[y, x]`. |
| Confiança no tracking | Detector recebe limiar 0.10; estágio principal começa em 0.25; novas trajetórias exigem 0.35. Esses valores pertencem ao experimento simples, não são defaults universais. |
| Identidade | `-1` indica ausência de ID confirmado; IDs são locais à execução e não são identidade global do veículo. |
| Aparência | O ByteTrack escolhido não usa embeddings, descritores de aparência ou compensação do movimento da câmera. |
| Reexecução | Tracker e histórico de rastros são reconstruídos no início do processamento do vídeo. |

## O que o material permite concluir

Os notebooks demonstram manipulação de resultados, consistência geométrica e construção de um fluxo temporal inspecionável. Os testes de conversão verificam o código geométrico; não medem a precisão das caixas do detector.

A inspeção dos vídeos é qualitativa. Resultados de contagem devem ser comparados com uma conferência manual explicitamente delimitada. Sem anotações temporais de referência, não se devem apresentar HOTA, IDF1 ou MOTA, nem tratar o total de IDs como número exato de veículos.
