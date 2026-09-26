# Fontes e atribuições

## RGBTDronePerson

Yan Zhang, Chang Xu, Wen Yang, Guangjun He, Huai Yu, Lei Yu e Gui-Song Xia.
*Drone-based RGBT tiny person detection*. ISPRS Journal of Photogrammetry and Remote Sensing,
204, 61–76, 2023. [Artigo](https://doi.org/10.1016/j.isprsjprs.2023.08.016).

A [página oficial](https://nnnnerd.github.io/RGBTDronePerson/) declara o dataset sob
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Os exemplos dos Notebooks 00, 01 e 02 incluem recortes, ampliações, espelhamento ilustrativo
e sobreposição de caixas. Essas transformações são identificadas nas explicações dos notebooks.
Os dados brutos são obtidos da distribuição oficial e não são armazenados neste repositório.

## VTUAV e vídeos da comparação

Pengyu Zhang, Jie Zhao, Dong Wang, Huchuan Lu e Xiang Ruan.
*Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline*. CVPR, 2022.
[Projeto oficial e downloads](https://zhang-pengyu.github.io/DUT-VTUAV/).

Os dois vídeos da comparação utilizam imagens da sequência `pedestrian_043` do **VTUAV**.
Foram aplicados amostragem temporal, redimensionamento, composição de painéis e sobreposição
de previsões produzidas pelo QFDet. As imagens ópticas e térmicas são dados reais do conjunto.
Usar o checkpoint RGBTDronePerson no primeiro vídeo não altera a origem VTUAV das imagens.
A figura `assets/video/primeiro-quadro-vtuav.png` foi extraída da saída já salva no Notebook 03:
é uma visualização térmica com previsões sobrepostas, da mesma sequência VTUAV.

A descrição oficial do VTUAV-det atribui o copyright das imagens à **School of Information
and Communication Engineering, Dalian University of Technology**. As páginas consultadas
disponibilizam o conjunto e solicitam citação, mas não explicitam uma licença de redistribuição
para os arquivos VTUAV. Não atribuímos a eles a licença CC BY 4.0 do RGBTDronePerson nem a
licença do código. Consulte as condições dos autores para novos usos ou redistribuição.

## Implementação QFDet

O [repositório dos autores](https://github.com/NNNNerd/mmdet-rgbtdroneperson) distribui a
implementação sob [Apache License 2.0](https://github.com/NNNNerd/mmdet-rgbtdroneperson/blob/main/LICENSE).
O script de preparação obtém o código diretamente desse repositório, preservando sua licença.
Os checkpoints também são obtidos pelos links mantidos pelos autores. A licença do código
não é estendida automaticamente aos datasets e checkpoints.

Os diagramas didáticos ilustram a arquitetura e o fluxo de inferência; as caixas e os valores
neles desenhados não representam medições. Os resultados reais são identificados nos notebooks.
