# Vídeos de apoio

Os arquivos desta pasta permitem observar o comportamento antes de executar os notebooks. A identificação no nome segue `modelo_fonte_cena_finalidade.mp4`.

## Exemplos reproduzidos nos notebooks

- `RF-DETR-Medium_Roboflow-VEHICLES_tracking.mp4`: primeiros oito segundos do [vídeo oficial VEHICLES](https://storage.googleapis.com/com-roboflow-marketing/supervision/cookbooks/vehicles-1280x720.mp4), com RF-DETR Medium, ByteTrack e Supervision. A fonte original tem 25 FPS. O notebook 02 reproduz a inferência e o tracking.
- `YOLOv8x_Roboflow-Traffic-Analysis_contagem.mp4`: prévia 960 × 540 do resultado do notebook 03. Vídeo e checkpoint são os fornecidos no [exemplo oficial Traffic Analysis](https://github.com/roboflow/supervision/tree/develop/examples/traffic_analysis). O processamento preserva os 806 quadros da entrada; a execução também produz a saída completa em 1920 × 1080.

Modelos e bibliotecas da execução atual estão descritos em [VALIDACAO.md](../../VALIDACAO.md). As fontes e os hashes dos downloads estão nos próprios notebooks.

## Casos complementares VisDrone

Os arquivos abaixo preservam exemplos de **detecção e rastreamento sem contagem**. O detector usado foi RF-DETR Medium com pesos COCO, sem treinamento neste projeto, associado ao pacote Trackers. A execução original utilizou `rfdetr==1.10.1`, `supervision==0.30.5` e `trackers==2.6.0`; as versões da prática atual são explicitadas separadamente.

| Cena | Recorte | Finalidade |
|:--|:--|:--|
| `uav0000117_02622_v` | Quadros de origem 1–300. | Cena noturna, oclusões e mudanças de iluminação/pose da câmera. |
| `uav0000305_00000_v` | Quadros de origem 1–184. | Cruzamento zenital com movimento de câmera e ambiguidades de associação. |

Fonte: [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset), obtido pelo [espelho Voxel51/visdrone-mot](https://huggingface.co/datasets/Voxel51/visdrone-mot/tree/3330e2096cf866ec0a70ae663681b6a5c8a967d9), na revisão fixa `3330e2096cf866ec0a70ae663681b6a5c8a967d9`.

**Os 25 FPS destes dois MP4s são uma taxa de reprodução escolhida.** A taxa temporal efetiva dos quadros no espelho não foi identificada. Portanto, a duração dos vídeos não permite inferir velocidade física dos veículos. A cena zenital recebeu uma linha de padding na base ao codificar a altura originalmente ímpar, sem deslocar a origem das coordenadas.

Os vídeos mostram predições, não caixas ou IDs de referência. Examine perdas de detecção, mudanças de ID e rastros durante o movimento da câmera. A licença das bibliotecas de processamento não substitui as condições dos vídeos e datasets de origem.

[MANIFESTO.json](MANIFESTO.json) registra os arquivos efetivamente entregues, suas dimensões, taxas de reprodução e hashes SHA-256.
