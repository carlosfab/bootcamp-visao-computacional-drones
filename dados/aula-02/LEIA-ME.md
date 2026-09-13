Imagem sintética gerada por IA em 13/09/2026.

Cena fictícia de uma praça, criada para a demonstração de caixas, recortes e inferência no notebook de detecção. Arquivo RGB de 1536 × 1024 pixels. Os notebooks de detecção e saídas usam uma cópia de 768 × 512 pixels; o de mapas de características usa 384 × 256 pixels.

Não é fotografia de voo, conjunto de treinamento ou benchmark. As caixas manuais do notebook são exemplos aproximados; as caixas do detector são previsões, não anotações de referência.

`formatos-caixas.png`: ilustração gerada por IA em 13/09/2026 para comparar `xyxy`, `xywh` e `cxcywh`. As três representações descrevem a mesma caixa, com valores escolhidos manualmente. A composição é esquemática, sem escala métrica, e não apresenta uma previsão de modelo.

As convenções são explicadas em [Dive into Deep Learning, seção 14.3.1](https://d2l.ai/chapter_computer-vision/bounding-box.html#bounding-boxes), [Foundations of Computer Vision, seção 50.4.1](https://visionbook.mit.edu/object_recognition_v3.html) e na [documentação de box_convert do TorchVision](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.box_convert.html). A ilustração foi criada para esta aula; não reproduz figuras dessas fontes.

`cnn-classificacao.png`: ilustração esquemática gerada por IA em 13/09/2026 para o notebook de treinamento. Representa a sequência Conv2d → ReLU → MaxPool2d → Flatten → Linear. As dimensões por imagem são 1 × 28 × 28 → 8 × 28 × 28 → 8 × 14 × 14 → 1.568 → 10. As texturas não representam ativações medidas.

As imagens de roupas do notebook 04 vêm do [Fashion-MNIST oficial, da Zalando Research](https://github.com/zalandoresearch/fashion-mnist). O auxiliar baixa os arquivos originais por HTTPS e verifica os checksums definidos pelo TorchVision. O dataset não é hospedado neste repositório.
