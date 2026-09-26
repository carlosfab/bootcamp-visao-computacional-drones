# Projeto 3 · Detecção, rastreamento e contagem de veículos

Este projeto parte das previsões de um detector e constrói um sistema de análise temporal. O objetivo é compreender as representações e as decisões geométricas antes de interpretar o vídeo anotado como uma medição.

Comece por [FUNDAMENTOS.md](FUNDAMENTOS.md), que reúne dez infográficos numerados sobre detecção, gradientes, descritores, associação, ByteTrack e coordenadas. O [índice visual](assets/infograficos/README.md) permite abrir cada figura, e o [roteiro de narração](ROTEIRO_VOICE_OVER.md) acompanha a mesma ordem, com falas técnicas e indicações de apresentação. Em seguida, execute os notebooks na ordem abaixo. Cada um é independente e contém seus próprios downloads e preparação.

| Material | Conteúdo | Resultado |
|:--|:--|:--|
| [01_supervision.ipynb](01_supervision.ipynb) | `Detections`, adaptadores, filtros, caixas, normalização, resize, padding, âncoras e anotadores. | Figuras que tornam explícitas as representações e suas transformações. |
| [02_tracking.ipynb](02_tracking.ipynb) | RF-DETR Medium, ByteTrack, identidades e trajetórias. | Oito segundos de vídeo processados em sequência, sem contador. |
| [03_projeto_final.ipynb](03_projeto_final.ipynb) | YOLOv8x especializado, tracking, cruzamento de linha e verificação. | Vídeo completo da rotatória, eventos CSV e protocolo de contagem manual. |

Os notebooks entregues incluem outputs de uma execução real. A qualidade das associações deve ser examinada separadamente da ausência de erros de código.

## Executar no Google Colab

1. Baixe o notebook desejado e abra o [Google Colab](https://colab.research.google.com/).
2. Use **Arquivo → Fazer upload de notebook**. Não é necessário publicar este branch ou enviar os vídeos de entrada manualmente.
3. Selecione uma GPU em **Ambiente de execução → Alterar tipo de ambiente de execução**. Uma T4 é suficiente para os modelos usados; a disponibilidade depende do serviço.
4. Use um ambiente novo e execute a instalação. Se o runtime já tiver importado bibliotecas que serão atualizadas, reinicie a sessão após instalar e execute novamente desde o início.
5. Execute as células em ordem e observe as saídas intermediárias. No projeto final, o ponto de interrupção para anotar uma referência manual está explicitamente indicado antes da inferência.
6. Salve sua cópia do notebook e baixe os arquivos de `resultados/` antes de encerrar o runtime. O armazenamento temporário não é permanente.

O código usa inferência local. Não é necessário obter uma chave da Roboflow, contratar uma API de inferência ou treinar um detector. Os downloads vêm de fontes públicas identificadas, com verificação de integridade.

## Ambiente local de referência

A validação foi realizada com **Python 3.12 em Linux, GPU NVIDIA e CUDA 12.8**, em ambiente separado dos projetos anteriores. As bibliotecas e as versões fixadas estão em [requirements.txt](requirements.txt). Não instale este conjunto sobre um ambiente que dependa de NumPy 1.x.

No terminal, entre na pasta deste projeto e prepare um ambiente virtual:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt jupyterlab
python -m jupyterlab
```

Os notebooks também selecionam CPU quando CUDA não está disponível, com maior tempo de inferência. A execução integral em CPU e as combinações de pacotes para Windows/macOS não fazem parte da validação registrada. Consulte [VALIDACAO.md](VALIDACAO.md) para o ambiente efetivamente testado e seus limites.

Os notebooks 01 e 02 usam um vídeo de aproximadamente 2,8 MB; o projeto final baixa um vídeo de aproximadamente 349 MB e um checkpoint de aproximadamente 137 MB. Reserve espaço adicional para bibliotecas, pesos do RF-DETR e vídeos de saída. Em uma reexecução, arquivos de entrada íntegros podem ser reutilizados.

## Exemplos prontos para observar

| Cena | Modelo e fonte | Uso |
|:--|:--|:--|
| [Rodovia](assets/videos/RF-DETR-Medium_Roboflow-VEHICLES_tracking.mp4) | RF-DETR Medium; vídeo Roboflow VEHICLES. | Resultado reproduzido pelo notebook 02, com caixas, IDs e rastros. |
| [Rotatória](assets/videos/YOLOv8x_Roboflow-Traffic-Analysis_contagem.mp4) | YOLOv8x especializado; exemplo Roboflow Traffic Analysis. | Prévia do resultado do notebook 03, com contagem no acesso direito. |
| [Cruzamento noturno](assets/videos/RF-DETR-Medium_VisDrone-MOT_uav0000117_02622_v_tracking.mp4) | RF-DETR Medium; VisDrone-MOT. | Observação de detecção e tracking, sem contador. |
| [Cruzamento zenital](assets/videos/RF-DETR-Medium_VisDrone-MOT_uav0000305_00000_v_tracking.mp4) | RF-DETR Medium; VisDrone-MOT. | Discussão de movimento da câmera e identidade, sem contador. |

O vídeo da rodovia contém os oito segundos completos do notebook 02. O da rotatória é uma prévia em resolução reduzida; o notebook 03 gera também o arquivo completo em 1080p. Os dois VisDrone são exemplos complementares previamente processados, com modelos, versões e recortes documentados em [assets/videos/README.md](assets/videos/README.md). Eles não são reexecutados pelos três notebooks e não constituem um benchmark.

Os nomes identificam **modelo e fonte do vídeo**, não necessariamente o conjunto usado no treinamento do detector. RF-DETR Medium usa pesos COCO. O checkpoint `traffic_analysis.pt` do projeto final é especializado em veículos e registra treinamento anterior em `drone-8`; ele não é um checkpoint genérico COCO. A filmagem da rotatória é aérea; sua fonte não comprova o equipamento de captura.

## Como interpretar o projeto final

A regra de passagem usa os quatro cantos da caixa. Quando a caixa ainda atravessa a linha, não há um lado inequívoco. A `LineZone` também exige que todas as âncoras estejam na faixa delimitada pelas perpendiculares aos extremos do segmento. Uma observação fora dessa faixa é desconsiderada, sem apagar automaticamente o histórico da trajetória. Isso evita interpretar a linha como uma fronteira infinita e reduz eventos causados por pequenas oscilações de um veículo parado sobre ela.

Um resultado de contagem precisa ser conferido por evento: direção, instante e veículo. **Totais iguais podem esconder omissões e duplicações que se cancelam.** A quantidade de IDs distintos também não equivale automaticamente à quantidade de veículos únicos.

O notebook fornece um CSV manual vazio e um protocolo de anotação independente. Nenhuma célula preenche essa referência com predições. A ausência de referência impede afirmar precisão de contagem ou métricas de identidade.

## Entrega do aluno

Entregue o notebook final executado, o vídeo anotado, `eventos.csv`, `serie_temporal.csv`, `resumo.json`, `ambiente.json` e a anotação manual. Acrescente uma análise curta que diferencie erros do detector, erros de associação e erros do critério de passagem, apoiada em quadros ou eventos concretos.

A [bibliografia e atualização técnica](REFERENCIAS.md) registra as fontes consultadas em 26/09/2026. Os artigos do blog fornecem contexto; os exemplos foram adaptados e verificados contra a API das versões fixadas.
