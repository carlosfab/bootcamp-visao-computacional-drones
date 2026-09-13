# Aula 2 · PyTorch e visão computacional

| Ordem | Notebook | Conteúdo | Executar |
|---|---|---|---|
| 01 | [Detecção de objetos](01-deteccao-objetos.ipynb) | Caixas, recortes, varredura e primeira inferência | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/01-deteccao-objetos.ipynb) |
| 02 | [Mapas de características](02-mapas-caracteristicas.ipynb) | Convolução, canais, ReLU e pooling | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/02-mapas-caracteristicas.ipynb) |
| 03 | [Saídas de um detector](03-saidas-detector.ipynb) | Limites dos recortes, previsões, formatos de caixas e visualização | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/03-saidas-detector.ipynb) |
| 04 | [Treinamento e inferência](04-treinamento-inferencia.ipynb) | Construção de uma CNN, treinamento, inferência e acurácia | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/04-treinamento-inferencia.ipynb) |

Execute a preparação e as demais células em ordem. Cada notebook baixa suas próprias imagens e funções auxiliares por URLs absolutas, fixadas em uma revisão do repositório. Não depende da execução dos anteriores.

GPU é opcional. A primeira execução dos notebooks de detecção baixa aproximadamente 167 MiB de pesos. A inferência em CPU pode levar mais tempo, especialmente na análise dos recortes do notebook 01.

A cena da praça é sintética. O notebook 04 usa o Fashion-MNIST: 12.000 imagens de treino, três épocas e acurácia no teste oficial. As imagens do dataset são baixadas automaticamente; GPU não é necessária para essa prática.

Os exemplos demonstram conceitos e interfaces; não constituem uma avaliação de modelos em voos reais.
