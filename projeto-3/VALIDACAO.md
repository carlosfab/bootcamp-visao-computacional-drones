# Validação da entrega

Data: **26/09/2026, UTC**. A versão atual foi reexecutada integralmente após o envio do branch ao GitHub: **70 de 70 células, sem erros**. Os arquivos foram obtidos diretamente do commit `52a1216b96ab34ecc1bd71da756502118f68fc42`, com diretório de dados e kernel novos para cada notebook. As fontes e os IDs das células foram comparados antes e depois da execução; as saídas entregues são as dessa reexecução.

## Reexecução da versão publicada

Ambiente: Linux x86_64, Python 3.12.3, GPU NVIDIA RTX 2000 Ada Generation de 16 GB, driver 570.195.03 e CUDA 12.8. As 12 dependências diretas fixadas em [requirements.txt](requirements.txt) foram conferidas, incluindo pandas 2.3.3; `pip check` não reportou conflitos.

| Notebook | Células executadas | Erros | Tempo registrado |
|:--|--:|--:|--:|
| 01_supervision.ipynb | 19 de 19 | 0 | 36,56 s |
| 02_tracking.ipynb | 18 de 18 | 0 | 40,18 s |
| 03_projeto_final.ipynb | 33 de 33 | 0 | 113,89 s |

Os tempos não incluem provisionamento da GPU e instalação inicial do ambiente. Não constituem estimativa garantida para Colab ou CPU. O Colab interativo e a execução integral em CPU continuam fora do escopo validado.

As novas células de download foram executadas com o vídeo oficial real. O notebook 02 gerou 200 quadros em H.264, 1280 × 720 e 25 FPS. O projeto final gerou 806 quadros em H.264, 1920 × 1080 e aproximadamente 29,97 FPS; seu processamento levou 46,98 s. A contagem permaneceu em **7 entradas e 9 saídas**, com 16 eventos e 806 registros temporais consistentes. Foram emitidos 129 IDs distintos, sem interpretar esse total como veículos únicos. A referência manual permaneceu vazia.

Foram conferidos 84 arquivos de retorno por tamanho e SHA-256, além dos 70 contadores de execução. As figuras de detecção, âncoras e seleção espacial e os quadros inicial, intermediário e final dos vídeos foram abertos para inspeção. A integração das saídas preservou integralmente o código, o texto e os IDs das células executadas. As quatro demonstrações em `assets/videos/` mantêm os arquivos e hashes de sua execução de origem, já documentados no manifesto.

## Histórico: ambiente da execução inicial

- Linux x86_64, Ubuntu 24.04, Python 3.12.3.
- GPU NVIDIA GeForce RTX 3090, 24 GB, driver 580.65.06, CUDA 12.8.
- Dependências diretas fixadas em [requirements.txt](requirements.txt), sem conflitos reportados por `pip check`.
- Pesos RF-DETR Medium reutilizados do cache íntegro entre execuções; dados de entrada baixados pelos notebooks para diretórios novos.

**O Colab interativo não foi validado nesta sessão:** a ferramenta de navegador não disponibilizou um navegador conectado. As instruções de upload e preparação estão incluídas, mas a execução comprovada é a do ambiente Linux acima. A execução integral em CPU também não foi medida.

## Histórico: execução inicial — 5b6fb20

| Notebook | Células de código executadas | Erros | Tempo da execução registrada |
|:--|--:|--:|--:|
| 01_supervision.ipynb | 18 de 18 | 0 | 16,46 s |
| 02_tracking.ipynb | 17 de 17 | 0 | 22,38 s |
| 03_projeto_final.ipynb | 29 de 29 | 0 | 80,42 s |

**Total: 64 células, sem erros.** Esses tempos pertencem à máquina e ao estado de instalação/cache descritos. Não incluem o provisionamento e a instalação inicial completa do ambiente, nem constituem estimativa garantida para Colab ou CPU.

O notebook 01 verificou numericamente conversões de caixas, normalização e transformações de escala/padding. Suas imagens de detecção, âncoras e seleção espacial foram inspecionadas.

O notebook 02 processou 200 quadros consecutivos, equivalentes a oito segundos a 25 FPS. A saída H.264 mantém 1280 × 720 pixels. Foram inspecionados os quadros inicial, intermediário e final.

O notebook 03 processou 806 quadros em 1920 × 1080, a aproximadamente 29,97 FPS. O processamento do vídeo levou 32,63 s nessa GPU. O vídeo H.264 completo preserva contagem de quadros, resolução e taxa de reprodução; as prévias incorporadas têm 960 × 540.

## Resultado de contagem e seu alcance

A execução registrada produziu **7 entradas e 9 saídas** no acesso direito da rotatória, usando os quatro cantos das caixas. Os 16 eventos do CSV concordam com os contadores do vídeo e com os 806 registros da série temporal.

Os **129 IDs confirmados distintos** não são apresentados como 129 veículos únicos. Fragmentações ou associações incorretas podem alterar esse número. Também não se afirma que a contagem 7/9 seja uma referência humana exaustiva: o CSV manual permanece vazio para ser preenchido de forma independente.

O teste de oscilação sobre a linha verifica uma propriedade da regra geométrica. Ele não mede precisão de detecção, tracking ou contagem no mundo real. A avaliação de qualidade exige inspeção e correspondência por evento, conforme o protocolo do notebook.

## Inspeção e revisão

Na entrega inicial, um revisor independente examinou o código, as fontes, os outputs, os fundamentos e os sete infográficos originais. A revisão conferiu a implementação atual de ByteTrack, os sistemas de coordenadas, as regras de estado e a ausência de caminhos privados ou credenciais nas células. As correções de notação e terminologia foram incorporadas.

Na ampliação de 26/09/2026, foram acrescentados três infográficos — derivadas/gradiente, SIFT/SURF/ORB e SORT/Deep SORT/ByteTrack — e um [roteiro de narração](ROTEIRO_VOICE_OVER.md) com dez blocos. Um revisor independente inspecionou individualmente as novas imagens, as fontes e a correspondência com as falas, sem apontamentos de prioridade P1/P2 nessa revisão. As durações são estimadas por contagem de palavras e pausas; não houve gravação de áudio.

As figuras dos notebooks foram abertas, assim como os quadros de início, meio e fim dos vídeos. Casos de oclusão, veículos distantes e sobreposição de rótulos permanecem visíveis para discussão; os resultados não foram editados para ocultar falhas.

Os dez infográficos foram gerados integralmente e preservados na resolução nativa de 1672 × 941 pixels, aproximadamente 16:9. A proporção difere em 0,0531% do formato exato; não foram aplicados recorte ou redimensionamento procedural. Os sete originais foram preservados byte a byte, com renumeração para a sequência 01–10. A numeração aparece nos arquivos, no índice e no roteiro; as imagens mantêm a composição sem cabeçalho ou rodapé recorrente. Dimensões e hashes estão em [assets/infograficos/MANIFESTO.json](assets/infograficos/MANIFESTO.json).

Na ampliação visual `5e5aa25`, os três notebooks, as dependências e os resultados executáveis permaneceram idênticos à entrega `5b6fb20`; as mudanças foram restritas a imagens e textos de apoio.

Os vídeos desta entrega têm hashes e metadados em [assets/videos/MANIFESTO.json](assets/videos/MANIFESTO.json). As fontes dos vídeos, pesos e referências técnicas estão documentadas nos notebooks e em [REFERENCIAS.md](REFERENCIAS.md).

## Preparação para merge

Os notebooks 01 e 02 passaram a baixar o vídeo para um arquivo temporário, conferir o SHA-256 e só então substituir o destino. A reexecução pode recuperar um arquivo incompleto; um arquivo íntegro é reutilizado. Testes direcionados verificaram primeiro download, cache, destino incompleto, interrupção, hash divergente e recuperação em ambos os notebooks, totalizando 12 casos.

A descrição do ambiente foi corrigida para distinguir versões instaladas de módulos já carregados. A documentação da `LineZone` explicita a faixa finita das âncoras. A validação do CSV manual foi revista para não descartar silenciosamente entradas com acentos, caixa ou espaços diferentes e para interromper a comparação diante de valores inválidos.

A validação direcionada do CSV incluiu 34 verificações: categorias canônicas, acentos/espaços/caixa, desconhecidos, campos inválidos, limites temporais e referência vazia. O limite inclusivo de um quadro tem tolerância aritmética de `1e-9` segundo para não rejeitar valores válidos por arredondamento de ponto flutuante. Somadas aos 12 casos de download, são **46 verificações direcionadas**. A revisão independente das correções não deixou pendências; a execução integral de 70 células acima cobre o fluxo completo da versão publicada.
