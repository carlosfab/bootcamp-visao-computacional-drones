# Validação da entrega

Data: **26/09/2026, UTC**. Os três notebooks foram executados integralmente, cada um em uma pasta de dados nova e em um kernel novo, com as células de instalação e downloads presentes no material do aluno. As células de código entregues foram comparadas com as efetivamente executadas.

## Ambiente testado

- Linux x86_64, Ubuntu 24.04, Python 3.12.3.
- GPU NVIDIA GeForce RTX 3090, 24 GB, driver 580.65.06, CUDA 12.8.
- Dependências diretas fixadas em [requirements.txt](requirements.txt), sem conflitos reportados por `pip check`.
- Pesos RF-DETR Medium reutilizados do cache íntegro entre execuções; dados de entrada baixados pelos notebooks para diretórios novos.

**O Colab interativo não foi validado nesta sessão:** a ferramenta de navegador não disponibilizou um navegador conectado. As instruções de upload e preparação estão incluídas, mas a execução comprovada é a do ambiente Linux acima. A execução integral em CPU também não foi medida.

## Execução

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

Um revisor independente examinou o código, as fontes, os outputs, os fundamentos e os sete infográficos. A revisão conferiu a implementação atual de ByteTrack, os sistemas de coordenadas, as regras de estado e a ausência de caminhos privados ou credenciais nas células. As correções de notação e terminologia foram incorporadas.

As figuras dos notebooks foram abertas, assim como os quadros de início, meio e fim dos vídeos. Casos de oclusão, veículos distantes e sobreposição de rótulos permanecem visíveis para discussão; os resultados não foram editados para ocultar falhas.

Os sete infográficos foram gerados integralmente e preservados na resolução nativa de 1672 × 941 pixels, aproximadamente 16:9. A proporção difere em 0,0531% do formato exato; não foram aplicados recorte ou redimensionamento procedural.

Os vídeos desta entrega têm hashes e metadados em [assets/videos/MANIFESTO.json](assets/videos/MANIFESTO.json). As fontes dos vídeos, pesos e referências técnicas estão documentadas nos notebooks e em [REFERENCIAS.md](REFERENCIAS.md).
