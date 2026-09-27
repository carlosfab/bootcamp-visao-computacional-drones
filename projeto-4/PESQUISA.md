# Escolha do projeto e dos dados

Pesquisa consultada em **27/09/2026**. A proposta é medir quantos bovinos estão
anotados em uma fotografia de drone e comparar três caminhos: detector genérico,
ajuste supervisionado e inferência em recortes. O ponto didático central é explicar
como dados, escala aparente e protocolo de avaliação alteram a resposta.

## Por que ICAERUS é o conjunto principal

O [ICAERUS v2, de Helary e Lebreton](https://zenodo.org/records/11048412), distribui
fotografias de pastagens, anotações e organização por fazenda e voo. Tem DOI
específico da versão e licença CC BY 4.0. Essa organização permite separar
capturas relacionadas e reservar uma fazenda para o teste. A resolução também
permite investigar objetos pequenos antes e depois da redução da imagem.

O ZIP original tem 16.586.848.207 bytes. Exigir seu download integral em cada aula
criaria uma dependência desnecessária. O projeto fornece uma seleção fixa de
300 imagens, redimensionadas proporcionalmente para lado máximo de 2.048 pixels,
em JPEG de qualidade 85. As anotações YOLO normalizadas são preservadas. Não há
rotação automática por EXIF, pois ela poderia desalinhá-las. O recorte é uma
adaptação didática; não é um benchmark oficial do ICAERUS.

O aluno recebe os arquivos prontos em partes ZIP menores que 40 MB. A extração
confere tamanho, SHA-256 de cada parte e de cada imagem/rótulo. Isso dispensa conta,
chave de API e conversão de formatos durante a aula. O script de reconstrução
acessa somente os membros selecionados do arquivo remoto por HTTP Range: cerca
de 3,4 GB de transferência, em vez de baixar o ZIP completo.

## Auditoria própria das anotações

Os números abaixo resultam da leitura do diretório do ZIP e de seus arquivos de
anotação, e não de métricas de um modelo:

| Verificação | Resultado |
|---|---:|
| Imagens originais no inventário | 1.385 |
| Voos, identificados por fazenda e diretório de captura | 25 |
| Imagens com pelo menos uma caixa YOLO | 366 |
| Imagens com TXT presente e vazio | 1.019 |
| Imagens sem um TXT correspondente | 0 |
| Correspondências de TXT ambíguas | 0 |
| Caixas efetivamente encontradas nos TXT | 4.747 |
| Classes efetivamente usadas nos TXT | somente índice 0, `cow` |

A página de origem anuncia **4.941 caixas**. A contagem observada nos TXT é
**4.747**; não tentamos completar a diferença com anotações inventadas. Alguns
arquivos de nomes declaram `calf`, mas não encontramos índice 1 nas anotações
YOLO. O núcleo usa uma única classe `cow`.

Uma imagem, `DJI_20230726084333_0004_V`, tem 14 caixas no TXT e 15 objetos no XML
correspondente; o objeto adicional recebe o nome `red_squirrel`. Ela foi excluída
da seleção. Isso registra uma inconsistência de formatos; não demonstra que todos
os demais rótulos sejam visualmente perfeitos. A auditoria estrutural não substitui
a revisão humana das imagens.

Um TXT ausente **não** seria tratado como imagem negativa. Aqui as imagens
negativas têm um arquivo de anotação correspondente, presente e vazio. Mesmo assim,
“negativa” significa sem bovino anotado, não uma garantia absoluta de que nenhum
bovino esteja visível.

## Partições fixas

| Partição | Imagens | Positivas | Negativas | Caixas | Origem |
|---|---:|---:|---:|---:|---|
| Treino | 180 | 120 | 60 | 1.480 | Mauron e Jalogny, antes de setembro de 2023 |
| Validação | 60 | 18 | 42 | 278 | Mauron e Jalogny, a partir de setembro de 2023 |
| Teste | 60 | 20 | 40 | 111 | Derval, fazenda reservada |
| **Total** | **300** | **158** | **142** | **1.869** | |

A seleção usa a semente textual `sigmoidal-gado-v1`, ordenação pelo SHA-256 do
caminho original e rodízio entre grupos de fazenda/voo, separadamente para positivas
e negativas. O manifesto de origem contém todos os caminhos, offsets do ZIP,
rótulos, CRC32 e a seleção final. Não há sorteio ao abrir um notebook.

O treino foi enriquecido em imagens positivas: seu universo elegível contém
130 positivas em 739 imagens, e selecionamos 120 positivas. Essa escolha fornece
mais exemplos úteis em um treino curto, mas altera a prevalência. Na validação,
a origem elegível tem 132 positivas em 459 imagens; em Derval, 41 em 124. As
seleções de 18/60 e 20/60 mantêm proporções próximas às respectivas origens.
As métricas ainda descrevem este recorte, não a distribuição de todas as pastagens.

Nenhum voo nem par fazenda/data atravessa as partições. Derval não participa do
ajuste de pesos ou de limiares. Isso torna a avaliação mais informativa que uma
divisão aleatória de fotografias consecutivas. Entretanto, são apenas 60 imagens
de teste, 20 positivas, dois voos e uma fazenda: não permitem prometer desempenho
em novas regiões, raças ou estações. Os animais não têm identidade anotada; não
afirmamos que indivíduos do treino e da validação sejam diferentes.

## O que a literatura muda no desenho da aula

[Shao et al.](https://doi.org/10.1080/01431161.2019.1624858) estudam detecção e
contagem com imagens aéreas e registram dificuldades como truncamento, desfoque
e oclusão. A atividade pede exemplos reais dessas falhas quando presentes, em vez
de reduzir a discussão a uma única média.

O trabalho [*Cattle counting in the wild with geolocated aerial images in large
pasture areas*](https://www.sciencedirect.com/science/article/abs/pii/S0168169921003719)
trata a remoção de animais duplicados entre imagens sobrepostas como uma etapa
própria. Por isso este projeto não soma fotografias para afirmar o tamanho de um
rebanho. A deduplicação ensinada no núcleo reúne recortes **da mesma fotografia**;
não resolve associação geográfica ou temporal entre fotografias.

O artigo [SAHI](https://arxiv.org/abs/2202.06934) fundamenta o uso de recortes para
objetos pequenos. O núcleo expõe explicitamente offsets, sobreposição e supressão
de caixas para que o aluno consiga inspecionar cada passo. Trata-se de uma
implementação didática de inferência em recortes, não uma reprodução integral do
framework ou dos resultados do artigo. Ganhos medidos em outros datasets não são
promessas para bovinos.

O próprio [repositório ICAERUS](https://github.com/ICAERUS-EU/UC3_Livestock_Monitoring)
inclui um modelo de bovinos para imagens grandes divididas em partes. Ele serve
como referência aplicada. Seus pesos não substituem o ajuste do aluno no
experimento principal: sua exposição aos dados precisaria ser investigada antes
de compará-los como um baseline independente.

## Alternativas investigadas

| Fonte | Vantagem didática | Motivo da decisão |
|---|---|---|
| [Roboflow 100 — aerial-cows](https://universe.roboflow.com/roboflow-100/aerial-cows) | Uma classe e acesso simples; o [espelho Francesco](https://huggingface.co/datasets/Francesco/aerial-cows/tree/main) tem arquivo de cerca de 144 MB | Alternativa leve. Seus metadados registram redimensionamento por esticamento a 640 × 640; isso limita o estudo de detalhe original e não oferece a mesma separação explícita por fazenda/voo |
| [Verschoor Aerial Cow Dataset](https://isis-data.science.uva.nl/jvgemert/conservationDronesECCV14w/) | Vídeos com IDs de trajetória e campos de oclusão | Boa extensão de rastreamento. Acrescentaria decodificação temporal e decisões sobre quadros correlacionados ao núcleo |
| [CattleEyeView](https://github.com/AnimalEyeQ/CattleEyeView) | Tarefas de detecção, contagem, pose e rastreamento | Extensão futura; o acesso oficial aponta para formulário, criando uma dependência de cadastro na reprodução imediata |
| [Counting Cows, imagens de satélite](https://arxiv.org/abs/2011.07369) | Problema aplicado e discussão de contagem em grandes áreas | Outro sensor e outro domínio; não substitui o conjunto de drone |

Os [metadados fixados do espelho RF100](https://huggingface.co/datasets/Francesco/aerial-cows/raw/5f996521a6bfdd65fb2a0fc5fe1ffcbb4207f70e/dataset_info.json)
registram 1.723 imagens: 1.084/299/340 nas partições originais, COCO e CC BY 4.0.
O arquivo indicado tem 143.972.570 bytes. A revisão consultada foi
`5f996521a6bfdd65fb2a0fc5fe1ffcbb4207f70e`. Essa alternativa foi pesquisada, mas
não foi misturada silenciosamente ao ICAERUS nem validada como substituição direta
dos notebooks. Trocar a fonte exige nova versão de dados e novos resultados.

Blogs oficiais da Roboflow e a documentação Ultralytics ajudam com a aplicação
das ferramentas; artigos e repositórios originais sustentam as decisões técnicas.
Não adotamos números promocionais de acurácia como evidência deste projeto.

## Limites assumidos

A redução a 2.048 pixels economiza armazenamento, mas já elimina detalhes do
original. Tiling pode preservar detalhe em relação a uma entrada menor do detector;
não recupera pixels perdidos na preparação. O efeito deve ser medido.

Uma semente não garante igualdade bit a bit entre GPUs e versões de PyTorch.
O compromisso é congelar dados e configurações, registrar ambiente e checkpoints
e tornar a comparação repetível. Os resultados e tempos de referência devem vir
de execução registrada; este documento de pesquisa não declara uma acurácia ainda
não medida.
