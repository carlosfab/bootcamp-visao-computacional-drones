# Referências do Projeto 4

Consultadas em 27/09/2026. Os links abaixo apontam para autores, mantenedores ou
distribuidores originais. Os achados da auditoria local estão em `PESQUISA.md` e
nos manifestos; não devem ser atribuídos como resultados dos artigos.

## Dados e aplicação em pecuária

1. **Helary, L.; Lebreton, A. (2024).** *Drone images and their annotations of
   grazing cows*, v2. Institut de l’Elevage / ICAERUS, Zenodo.
   [Registro e DOI](https://doi.org/10.5281/zenodo.11048412).
   Fonte das imagens e dos rótulos. A licença `cc-by-4.0` também é verificável nos
   [metadados públicos](https://zenodo.org/api/records/11048412).

2. **ICAERUS / Institut de l’Elevage.** *UC3 Livestock Monitoring*.
   [Repositório oficial](https://github.com/ICAERUS-EU/UC3_Livestock_Monitoring) e
   [modelo para imagens grandes](https://www.platform.icaerus.eu/models-algorithms/details/Cow%20detection%20model%20v2/11).
   Referência aplicada para reconhecimento de bovinos em imagens divididas em partes.

3. **Shao, W. et al. (2020; publicação online em 2019).** *Cattle detection and
   counting in UAV images based on convolutional neural networks*.
   International Journal of Remote Sensing, 41(1), 31–52.
   [DOI 10.1080/01431161.2019.1624858](https://doi.org/10.1080/01431161.2019.1624858).
   Fundamenta a discussão sobre detecção aérea, contagem, oclusão e casos de borda.

4. **Cattle counting in the wild with geolocated aerial images in large pasture
   areas (2021).** Computers and Electronics in Agriculture.
   [Página do artigo](https://www.sciencedirect.com/science/article/abs/pii/S0168169921003719).
   Referência para distinguir detecção por imagem e remoção de duplicatas entre
   imagens de uma área. A síntese do projeto usa o resumo público disponível.

5. **Counting cattle in UAV images using convolutional neural network (2023).**
   Remote Sensing Applications: Society and Environment, 29, 100900.
   [DOI 10.1016/j.rsase.2022.100900](https://doi.org/10.1016/j.rsase.2022.100900).
   Contextualiza a comparação de detectores em imagens de gado Nelore; não
   transportamos suas métricas para o recorte ICAERUS.

## Objetos pequenos e ferramentas

6. **Akyon, F. C.; Altinuc, S. O.; Temizel, A. (2022).** *Slicing Aided Hyper
   Inference and Fine-tuning for Small Object Detection*.
   [Artigo](https://arxiv.org/abs/2202.06934) e
   [implementação oficial SAHI](https://github.com/Small-Object-Detection/SAHI).
   Base conceitual para a inferência em recortes. O código didático do projeto
   expõe a projeção das caixas e NMS; não implementa todas as opções do framework.

7. **Ultralytics.** [Guia oficial de inferência em recortes com SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference/),
   [treinamento](https://docs.ultralytics.com/modes/train/) e
   [validação](https://docs.ultralytics.com/modes/val/).
   Documentação operacional. As páginas evoluem; a versão de execução do projeto
   é fixada no arquivo `requirements.txt`.

8. **Roboflow.** [How to Detect Small Objects in Drone Imagery with RF-DETR](https://blog.roboflow.com/how-to-detect-small-objects-in-drone-imagery/)
   e [How to Count Objects in a Zone](https://blog.roboflow.com/how-to-count-objects-in-a-zone/).
   Tutoriais dos mantenedores para contexto aplicado. Contagem em zona ou por
   quadro não é evidência de contagem de indivíduos únicos em uma fazenda.

## Conjuntos alternativos

9. **Ciaglia, F. et al. (2022).** *Roboflow 100: A Rich, Multi-Domain Object
   Detection Benchmark*. [Artigo](https://arxiv.org/abs/2211.13523),
   [aerial-cows no Roboflow](https://universe.roboflow.com/roboflow-100/aerial-cows) e
   [espelho Francesco no Hugging Face](https://huggingface.co/datasets/Francesco/aerial-cows).
   O [arquivo de metadados na revisão consultada](https://huggingface.co/datasets/Francesco/aerial-cows/raw/5f996521a6bfdd65fb2a0fc5fe1ffcbb4207f70e/dataset_info.json)
   permite conferir classe, licença, contagens e o redimensionamento a 640 × 640.
   A página do Roboflow atribui o conjunto original a Omar Kapur, `wwblodge`,
   Ricardo Jenez, Justin Jeng e Jeffrey Day.

10. **Van Gemert, J. C. et al. (2014).** *Nature Conservation Drones for Automatic
    Localization and Counting of Animals*. ECCV workshop.
    [Verschoor Aerial Cow Dataset](https://isis-data.science.uva.nl/jvgemert/conservationDronesECCV14w/).
    Vídeos anotados com IDs de trajetórias e campos de oclusão, úteis para uma
    extensão temporal.

11. **Ong, K. E. et al. (2023).** *CattleEyeView: A Multi-task Top-down View Cattle
    Dataset for Smarter Precision Livestock Farming*. IEEE VCIP.
    [Artigo](https://arxiv.org/abs/2312.08764) e
    [repositório oficial](https://github.com/AnimalEyeQ/CattleEyeView).
    O acesso ao conjunto é indicado por formulário no repositório; não integra
    as dependências do núcleo da atividade.

12. **Counting Cows: Tracking Illegal Cattle Ranching From High-Resolution
    Satellite Imagery (2020).** [Artigo](https://arxiv.org/abs/2011.07369).
    Leitura opcional para comparar o problema de contagem em outro sensor e escala.

## Licença e atribuição dos dados distribuídos

As imagens e anotações do pacote são uma adaptação do ICAERUS v2, sob
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Foram selecionadas, separadas por grupos de captura, redimensionadas e
recomprimidas. O arquivo `ATRIBUICAO.md`, incluído nos pacotes, registra autores,
DOI, alterações e ausência de endosso. A licença dos dados não substitui a licença
das bibliotecas ou dos pesos utilizados nos experimentos.

## Licença Ultralytics e dos pesos de referência

A [cópia integral AGPL-3.0 distribuída com o projeto](assets/LICENSE-ultralytics.txt)
foi copiada, sem alterações, de `ultralytics-8.3.203.dist-info/licenses/LICENSE`
do pacote instalado **Ultralytics 8.3.203**. A versão, o campo de licença e a origem
GitHub foram conferidos nos metadados do pacote; o conteúdo confere com o SHA-256
registrado no arquivo `RECORD` da distribuição. A
[licença no tag oficial v8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/LICENSE)
permite consultar a fonte correspondente.

A cópia tem **34.523 bytes** e SHA-256
`0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
Ela acompanha os pesos YOLO e os checkpoints de referência do projeto na modalidade
aberta AGPL-3.0; os metadados de cada peso identificam seu arquivo e hash.
Os dados ICAERUS permanecem sob CC BY 4.0, com atribuição própria.
