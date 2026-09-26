# Detecção, representação e rastreamento de objetos

O problema central é transformar uma sequência de imagens em hipóteses sobre **onde estão os objetos, a que classes pertencem e quais observações correspondem ao mesmo objeto ao longo do tempo**. Nesta aula, veículos em imagens aéreas fornecem o contexto; a formulação também se aplica a outras cenas.

As ilustrações são esquemáticas. Cores, setas, vetores e caixas explicam relações conceituais; não representam medições de um experimento. As fontes primárias e as APIs utilizadas estão reunidas em [REFERENCIAS.md](REFERENCIAS.md).

## Detecção e identidade temporal

![Dois veículos detectados em três quadros; no rastreamento, cada veículo mantém seu identificador e a correspondência entre observações.](assets/infograficos/01-deteccao-rastreamento.png)

Um detector produz um conjunto de hipóteses por imagem. Cada hipótese pode incluir uma caixa delimitadora, uma classe e um escore de confiança. A classe `car` informa uma categoria; ela não distingue individualmente todos os carros da cena. A posição de uma detecção na lista também não constitui identidade.

No **rastreamento de múltiplos objetos**, ou MOT, buscamos associar observações ao longo dos quadros. Um `tracker_id` identifica uma trajetória estimada dentro da execução. Ele não é uma placa, não garante identidade global e pode ser perdido, trocado ou fragmentado.

| Elemento | Pergunta respondida | Limite |
|:--|:--|:--|
| Caixa | Onde está a hipótese de objeto? | Localização estimada no plano da imagem. |
| Classe | A que categoria ela pertence? | Objetos diferentes podem compartilhar a categoria. |
| Confiança | Qual é o escore atribuído pelo detector? | Não certifica a associação temporal nem é necessariamente uma probabilidade calibrada. |
| ID de trajetória | A qual hipótese temporal a observação foi associada? | É local ao tracker e pode conter erros. |

**Questão para discussão:** se um carro desaparecer por alguns quadros e retornar com outro ID, quantos objetos físicos e quantas trajetórias terão sido observados?

## Pontos de interesse e descritores locais

![Um ponto selecionado sobre o veículo define uma vizinhança local, cuja aparência é representada por um vetor descritor.](assets/infograficos/02-pontos-descritores.png)

Um **detector de pontos de interesse** seleciona posições informativas da imagem, por exemplo regiões com variações locais de intensidade. Um **descritor** transforma a vizinhança de uma posição em uma representação comparável. Localizar um ponto e descrever sua aparência são operações relacionadas, mas distintas.

SIFT é uma referência clássica de localização em espaço de escalas e descrição por distribuições de gradientes. ORB combina pontos FAST orientados com um descritor baseado em comparações binárias de intensidade. A métrica de comparação deve respeitar a representação: descritores reais e descritores binários não são automaticamente comparáveis pela mesma regra. [Lowe, 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf); [OpenCV: ORB](https://docs.opencv.org/4.13.0/d1/d89/tutorial_py_orb.html).

Um carro pode conter vários pontos de interesse, e um ponto pode pertencer ao fundo. Correspondências locais também não são, isoladamente, trajetórias de objetos completos. A grade e as setas da ilustração indicam uma representação esquemática da aparência, sem especificar um descritor calculado ou sua dimensionalidade.

**Distinção adicional:** pontos semânticos de um modelo de pose, como articulações ou cantos de uma peça, têm significado definido pela tarefa. Eles não devem ser confundidos com pontos locais SIFT/ORB nem com IDs de tracking.

## Características aprendidas e aparência

![Dois recortes do mesmo veículo passam por um extrator aprendido e produzem vetores de aparência que podem ser comparados.](assets/infograficos/03-aparencia-aprendida.png)

Redes neurais aprendem representações a partir dos dados. Um mapa de características preserva organização espacial e canais; um vetor de aparência pode resumir um recorte de objeto para comparação. A arquitetura e a função de treinamento determinam o que essa representação tende a preservar.

Em métodos com **reidentificação**, ou Re-ID, a aparência fornece uma evidência adicional para associar observações. Deep SORT é uma referência de integração entre informação de movimento e uma métrica de aparência aprendida. Veículos semelhantes, mudanças de iluminação e oclusões continuam podendo tornar a correspondência ambígua. [Wojke et al., 2017](https://arxiv.org/abs/1703.07402).

Um vetor não é um ID pronto. A decisão de associação depende do método, dos candidatos e de seus critérios. Da mesma forma, as características internas de um detector não são automaticamente embeddings adequados à reidentificação.

**Nesta prática:** o detector usa uma rede neural, mas o `ByteTrackTracker` escolhido recebe caixas e escores. Seu associador não extrai descritores de aparência dos pixels. A imagem explica uma alternativa conceitual, não uma etapa oculta dos nossos notebooks.

## Associação geométrica e movimento

![Posições são previstas, caixas previstas e detectadas são comparadas por sobreposição, e correspondências um a um preservam IDs.](assets/infograficos/04-associacao-geometrica.png)

No paradigma **tracking-by-detection**, o rastreador recebe as hipóteses do detector. Um modelo de movimento prevê estados; a associação compara candidatos e atribui correspondências. A atribuição deve evitar que duas detecções atualizem simultaneamente a mesma trajetória sem uma regra explícita.

Para duas regiões $A$ e $B$, a interseção sobre união é

$$\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

A IoU mede sobreposição espacial, não identidade. Um veículo diferente pode ocupar uma posição semelhante após uma oclusão. O filtro de Kalman, por sua vez, estima estado e incerteza sob um modelo; ele não torna observável um objeto que desapareceu da imagem.

O exemplo visual mostra pares compatíveis, sem reproduzir uma matriz de custos completa. Na implementação usada, a atribuição é resolvida por um algoritmo de otimização; não corresponde a escolher independentemente o maior valor de cada linha. [SORT](https://arxiv.org/abs/1602.00763); [API ByteTrack](https://trackers.roboflow.com/latest/trackers/bytetrack/).

## ByteTrack: recuperar evidências fracas

![A associação de alta confiança resolve um veículo; uma segunda associação usa uma detecção fraca para recuperar o ID de uma van parcialmente ocluída.](assets/infograficos/05-bytetrack.png)

Uma detecção de baixa confiança pode corresponder a um objeto parcialmente ocluído. Descartar prematuramente todas essas detecções remove evidências que poderiam manter uma trajetória.

A ideia central de BYTE é associar primeiro as detecções de maior confiança e, depois, tentar recuperar trajetórias ainda sem correspondência usando detecções de menor confiança. Uma caixa fraca sem correspondência não deve iniciar uma nova trajetória nessa segunda etapa. [Zhang et al., 2022](https://arxiv.org/abs/2110.06864).

Há três decisões distintas nos notebooks: o limiar do detector, a separação entre as faixas de confiança e a condição para iniciar um novo track. Seus valores são parâmetros do experimento, não constantes universais. Aumentar o primeiro limiar acima da faixa fraca pode impedir que o segundo estágio receba qualquer candidato útil.

O estado do rastreador deve avançar em todos os quadros, inclusive quando não há detecções. A memória de uma trajetória ausente não implica que a biblioteca desenhe uma caixa predita naquele quadro. IDs negativos são filtrados após a atualização; o ID zero é válido.

## Coordenadas e significado geométrico

![A mesma caixa representada por dois cantos, por canto e dimensões e por centro e dimensões; em todos os casos, x cresce à direita e y para baixo.](assets/infograficos/06-coordenadas.png)

No sistema raster usado por OpenCV e pelos notebooks, a origem fica no canto superior esquerdo: $x$ cresce para a direita e $y$ para baixo. Um array de imagem é indexado como `imagem[y, x]`. Essa convenção deve ser declarada antes de converter coordenadas.

Para uma caixa $\mathbf{b}=(x_1,y_1,x_2,y_2)^\top$:

$$w=x_2-x_1,\qquad h=y_2-y_1,\qquad c_x=\frac{x_1+x_2}{2},\qquad c_y=\frac{y_1+y_2}{2}.$$

`xywh` usa canto superior esquerdo e dimensões; `cxcywh` usa centro e dimensões. Trocar os nomes sem transformar os valores desloca a caixa. Para largura $W$ e altura $H$, normalizamos coordenadas horizontais por $W$ e verticais por $H$. Normalização não corrige perspectiva nem converte pixels em metros.

Se houver redimensionamento e padding, a transformação deve ser aplicada às caixas e aos elementos geométricos da mesma cena. Com fatores $s_x,s_y$ e deslocamentos $p_x,p_y$, temos $x'=s_xx+p_x$ e $y'=s_yy+p_y$. A inversão exige remover o padding antes de dividir pela escala. Adaptadores podem já fornecer caixas na resolução original; transformar novamente é um erro.

Uma **âncora** escolhe o ponto ou conjunto de pontos usado em uma regra espacial. Centro, centro inferior e cantos não respondem à mesma pergunta. No projeto final, uma passagem só se estabelece quando os quatro cantos chegam ao outro lado da linha; enquanto a caixa atravessa a linha, ela não define um lado inequívoco. Essa regra reduz eventos causados por oscilação do centro de um veículo parado sobre a linha.

## Limitações e interpretação das saídas

![Quatro dificuldades: oclusão, poucos pixels por objeto, ambiguidade entre veículos semelhantes e deslocamento da cena causado pela câmera.](assets/infograficos/07-desafios.png)

Objetos pequenos fornecem poucos pixels; oclusões removem evidências; veículos semelhantes tornam associações ambíguas. O movimento da câmera altera simultaneamente muitas posições na imagem. Nenhum desses problemas é resolvido apenas pelo desenho de caixas ou pela atribuição de números.

Uma linha fixa em pixels pode deixar de representar a mesma seção da rua quando a câmera se move. Rastros na imagem também não equivalem a deslocamentos métricos no solo. Nos exemplos VisDrone complementares, observaremos detecção e tracking, sem contador e sem estimativa de velocidade.

Convém separar quatro verificações: **o código executou**, **as coordenadas são consistentes**, **as associações parecem plausíveis** e **o desempenho foi medido contra uma referência**. Uma não implica automaticamente a seguinte. Caixas bem desenhadas e totais coincidentes podem esconder erros de identidade, omissões e duplicações.

## Da teoria à implementação

Execute [01_supervision.ipynb](01_supervision.ipynb) para inspecionar representações e anotações; [02_tracking.ipynb](02_tracking.ipynb) para construir o fluxo temporal; e [03_projeto_final.ipynb](03_projeto_final.ipynb) para transformar trajetórias em eventos auditáveis.

Ao final, explique por que o número de objetos em um quadro, o número de IDs emitidos e o número de cruzamentos são grandezas distintas. Use exemplos concretos do vídeo para sustentar a resposta.
