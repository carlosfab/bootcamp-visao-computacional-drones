# Roteiro de narração: detecção e rastreamento de objetos

Leia apenas os parágrafos de **Fala**. Objetivos, indicações de foco e referências são orientações de apresentação. As transições já estão incorporadas ao final de cada fala. As imagens são esquemas explicativos, sem resultados quantitativos de um experimento.

**Duração estimada:** aproximadamente 13min50s–15min25s para as 1.917 palavras de fala, incluindo as pausas indicadas. A estimativa considera aproximadamente 130–145 palavras por minuto, mais quatro segundos de pausa por imagem. Não houve gravação nem medição de áudio; explicações adicionais e perguntas da turma aumentam a duração.

| Ordem | Infográfico | Duração estimada |
|:--:|:--|:--:|
| 01 | [Detecção e rastreamento](assets/infograficos/01-deteccao-rastreamento.png) | 1min10s–1min25s |
| 02 | [Derivadas e gradiente](assets/infograficos/02-derivadas-gradiente.png) | 1min50s–2min05s |
| 03 | [Pontos de interesse e descritores](assets/infograficos/03-pontos-descritores.png) | 1min10s–1min25s |
| 04 | [SIFT, SURF e ORB](assets/infograficos/04-sift-surf-orb.png) | 1min40s–1min55s |
| 05 | [Aparência aprendida](assets/infograficos/05-aparencia-aprendida.png) | 1min15s–1min25s |
| 06 | [Associação geométrica](assets/infograficos/06-associacao-geometrica.png) | 1min10s–1min25s |
| 07 | [SORT, Deep SORT e ByteTrack](assets/infograficos/07-sort-deepsort-bytetrack.png) | 1min20s–1min30s |
| 08 | [ByteTrack em duas etapas](assets/infograficos/08-bytetrack.png) | 1min10s–1min25s |
| 09 | [Coordenadas e âncoras](assets/infograficos/09-coordenadas.png) | 1min25s–1min40s |
| 10 | [Desafios e interpretação](assets/infograficos/10-desafios.png) | 1min15s–1min30s |

## 01. Detecção e rastreamento

**Imagem:** [01-deteccao-rastreamento.png](assets/infograficos/01-deteccao-rastreamento.png)

**Objetivo:** distinguir observações por imagem e hipóteses de identidade temporal.

**Indicação de apresentação:** acompanhar um veículo entre quadros; depois apontar a diferença entre classe e ID. Pausa de quatro segundos antes de avançar.

**Fala**

Considere uma sequência de imagens de veículos observados por uma câmera aérea. Em cada quadro, o detector tenta localizar objetos e atribuir categorias. Sua saída pode conter uma caixa, uma classe e um escore de confiança. Essas informações descrevem uma hipótese naquele instante.

Agora acompanhe o mesmo carro em dois quadros. A classe continua sendo carro, mas essa categoria também pertence aos demais veículos. Nem a classe nem a posição da caixa na lista de resultados informam, por si mesmas, que as duas observações correspondem ao mesmo objeto físico.

O rastreamento acrescenta essa pergunta temporal. Precisamos decidir quais observações atualizam trajetórias existentes, quais iniciam novas trajetórias e quais permaneceram sem correspondência. O identificador exibido sobre o veículo representa uma hipótese mantida pelo rastreador durante a execução.

Se um carro desaparece e retorna com outro identificador, temos um objeto físico e duas trajetórias estimadas. Portanto, contar identificadores não equivale automaticamente a contar veículos únicos. Para compreender as evidências usadas nessas decisões, começaremos pela informação presente na própria imagem.

## 02. Derivadas e gradiente da imagem

**Imagem:** [02-derivadas-gradiente.png](assets/infograficos/02-derivadas-gradiente.png)

**Objetivo:** explicar variação espacial de intensidade e separá-la do treinamento e do movimento.

**Indicação de apresentação:** percorrer a transição de intensidade e apontar a direção do gradiente. Manter a convenção raster: eixo horizontal para a direita e eixo vertical para baixo. Pausa de quatro segundos.

**Fala**

Uma imagem em tons de cinza associa uma intensidade a cada posição. A derivada horizontal descreve como essa intensidade varia quando deslocamos a posição horizontalmente. A derivada vertical descreve a variação na direção vertical. Como a imagem é discreta e pode conter ruído, estimamos essas derivadas por operações locais, frequentemente combinadas com suavização.

Reunindo as duas componentes, obtemos o gradiente espacial. Seu módulo indica a intensidade da variação local. Sua direção aponta para o aumento mais acentuado da intensidade. Próximo de uma borda regular, essa direção é perpendicular à borda; ela não informa para onde o veículo está se movendo. Na borda vertical ilustrada, a intensidade passa de escura, à esquerda, para clara, à direita. Nesse exemplo idealizado, o gradiente aponta horizontalmente para a direita e a componente vertical é nula. O perfil cresce mais rapidamente na transição, onde a derivada horizontal atinge seu pico positivo. Nos gráficos, o eixo vertical mostra valores das funções, não a coordenada vertical da imagem.

A orientação pode ser calculada com a função arco tangente de dois argumentos, usando primeiro a componente vertical e depois a horizontal. Nesta convenção raster, a coordenada vertical cresce para baixo. Por isso, os ângulos devem ser interpretados nesse mesmo sistema, sem inverter silenciosamente o sentido visual.

Também precisamos separar dois usos da palavra gradiente. Aqui derivamos a intensidade em relação à posição. No treinamento de uma rede, derivamos uma função de custo em relação aos parâmetros do modelo. São espaços e perguntas diferentes. O gradiente da imagem ajuda a representar estrutura local; sozinho, ele não resolve a correspondência temporal.

## 03. Pontos de interesse e descritores locais

**Imagem:** [03-pontos-descritores.png](assets/infograficos/03-pontos-descritores.png)

**Objetivo:** diferenciar seleção de posições, representação de vizinhanças e identidade de objetos.

**Indicação de apresentação:** apontar o ponto selecionado, ampliar sua vizinhança e acompanhar a passagem para o descritor. Pausa de quatro segundos.

**Fala**

Para comparar imagens, podemos selecionar regiões locais com estrutura informativa. Um detector de pontos de interesse procura posições que possam ser localizadas novamente quando a imagem sofre determinadas transformações. Cantos, regiões com contraste e estruturas observadas em diferentes escalas podem fornecer candidatos.

Depois da localização, precisamos descrever a vizinhança. O descritor transforma aquele conteúdo visual em uma representação comparável. Assim, a posição responde onde observar, enquanto o descritor resume o que foi observado ao redor dela.

Essa separação evita uma confusão frequente: detectar um ponto de interesse não é detectar um veículo. Um único carro pode conter vários pontos, e muitos pontos podem pertencer ao asfalto ou aos edifícios. Da mesma forma, dois descritores semelhantes indicam uma correspondência candidata, sem constituir uma identidade confirmada.

Na prática, localização repetível, representação discriminativa e uma comparação adequada precisam trabalhar em conjunto. Nenhuma dessas propriedades garante correspondência sob qualquer iluminação, escala ou perspectiva. SIFT, SURF e ORB tornam essas escolhas concretas e ajudam a entender como diferentes representações são construídas.

## 04. SIFT, SURF e ORB

**Imagem:** [04-sift-surf-orb.png](assets/infograficos/04-sift-surf-orb.png)

**Objetivo:** comparar mecanismos e representações de três métodos de características locais.

**Indicação de apresentação:** apresentar cada método por completo antes de comparar dimensionalidade e métrica. Pausa de quatro segundos.

**Fala**

SIFT, SURF e ORB são métodos de extração de características locais. A comparação deve considerar como encontram pontos, como atribuem orientação e como representam a vizinhança. Eles não são rastreadores de múltiplos objetos equivalentes ao ByteTrack.

No SIFT, a busca por pontos envolve extremos em um espaço de escalas construído com diferenças de imagens suavizadas por Gaussianas. Essa operação é conhecida como diferença de Gaussianas. Após localizar e orientar os pontos, o método organiza gradientes locais em histogramas. Na configuração clássica, quatro por quatro regiões, com oito orientações em cada uma, produzem cento e vinte e oito componentes.

O SURF usa uma aproximação do determinante da matriz Hessiana para localizar estruturas em diferentes escalas. Na descrição, agrega respostas de filtros de Haar nas direções horizontal e vertical. A representação básica tem sessenta e quatro componentes; a versão estendida tem cento e vinte e oito.

O ORB combina pontos FAST com orientação estimada localmente e uma versão orientada do descritor BRIEF. Na configuração usual, com dois pontos por teste, comparações de intensidade produzem um descritor binário de duzentos e cinquenta e seis bits.

A forma de comparar precisa acompanhar a representação. Para o ORB nessa configuração, usamos distância de Hamming, que conta diferenças entre bits. Não devemos tratar seus bytes como coordenadas comuns e aplicar distância euclidiana indiscriminadamente. Em SIFT e SURF, comparamos vetores numéricos de outra natureza. Agora veremos como uma representação também pode ser aprendida a partir dos dados.

## 05. Características aprendidas e aparência

**Imagem:** [05-aparencia-aprendida.png](assets/infograficos/05-aparencia-aprendida.png)

**Objetivo:** apresentar embeddings como evidências de aparência, sem confundi-los com IDs.

**Indicação de apresentação:** acompanhar os dois recortes até seus vetores e, depois, apontar a comparação. Pausa de quatro segundos.

**Fala**

Uma rede neural pode transformar um recorte de objeto em um vetor de características. Em vez de fixarmos todas as regras de descrição manualmente, ajustamos parâmetros usando exemplos e uma função de treinamento.

Quando o objetivo é reidentificação, procuramos uma representação que aproxime observações compatíveis com a mesma identidade e separe observações de identidades diferentes. Entretanto, o resultado depende dos dados, do objetivo de treinamento e da diferença entre o cenário de treinamento e o cenário de uso.

Observe que o vetor não contém um identificador pronto. Ele permite calcular uma medida de semelhança, que será combinada com os demais critérios do rastreador. Dois veículos muito parecidos ainda podem produzir uma situação ambígua.

Também não devemos assumir que qualquer característica interna de um detector seja adequada para reidentificação. Detectar a categoria carro e distinguir dois carros específicos são tarefas relacionadas, mas diferentes.

Nos nossos notebooks, o detector utiliza uma rede neural, enquanto o ByteTrack escolhido não extrai aparência dos recortes. Para compreender essa implementação, precisamos examinar a associação baseada em geometria e movimento.

## 06. Associação geométrica e movimento

**Imagem:** [06-associacao-geometrica.png](assets/infograficos/06-associacao-geometrica.png)

**Objetivo:** explicar previsão de estado, compatibilidade espacial e atribuição entre observações.

**Indicação de apresentação:** seguir a sequência previsão, comparação e associação; evitar apresentar a seta como uma trajetória garantida. Pausa de quatro segundos.

**Fala**

No rastreamento por detecção, o sistema recebe novas caixas e mantém um histórico de trajetórias. Um modelo de movimento permite prever onde cada trajetória poderá ser observada no próximo quadro. O filtro de Kalman é uma forma de estimar esse estado e sua incerteza sob hipóteses explícitas.

Em seguida, comparamos as previsões com as detecções atuais. Uma medida frequente é a interseção sobre união, ou IoU. Ela divide a área de interseção entre duas caixas pela área ocupada por sua união. Quanto maior a sobreposição, maior o valor dessa medida.

A IoU expressa compatibilidade espacial, não identidade. Um veículo diferente pode aparecer na posição prevista. Além disso, escolher o melhor candidato para cada trajetória isoladamente pode criar conflitos: duas trajetórias poderiam tentar usar a mesma detecção.

Por isso, a associação considera os candidatos em conjunto e impõe restrições de atribuição. Algumas trajetórias podem permanecer sem observação, e algumas detecções podem iniciar novos estados, conforme os critérios do método. Esse encadeamento permite comparar rastreadores que usam fontes de evidência diferentes.

## 07. SORT, Deep SORT e ByteTrack

**Imagem:** [07-sort-deepsort-bytetrack.png](assets/infograficos/07-sort-deepsort-bytetrack.png)

**Objetivo:** distinguir as evidências usadas por três abordagens de rastreamento por detecção.

**Indicação de apresentação:** comparar as colunas pelos critérios de associação, sem transformar a ordem em um ranking de desempenho. Pausa de quatro segundos.

**Fala**

SORT, Deep SORT e ByteTrack pertencem à discussão de associação temporal de objetos detectados. Diferentemente de SIFT, SURF e ORB, eles mantêm trajetórias e administram sua evolução entre quadros.

O SORT combina um modelo de movimento, estimado com filtro de Kalman, e uma associação baseada na geometria das caixas. Essa formulação é simples, mas pode perder correspondências quando há oclusões ou quando vários objetos disputam posições semelhantes.

O Deep SORT acrescenta uma métrica de aparência aprendida. A semelhança entre representações visuais fornece evidência adicional, junto à compatibilidade de movimento e às regras de associação. Isso não elimina ambiguidades, especialmente quando objetos distintos são visualmente semelhantes.

O ByteTrack enfatiza outra decisão: aproveitar detecções de menor confiança para recuperar trajetórias que não foram associadas na primeira etapa. Na implementação utilizada nesta aula, ele trabalha com caixas e escores, sem extrair descritores de aparência.

Essa comparação não estabelece que um método seja superior em qualquer cenário. A qualidade depende do detector, das hipóteses de movimento, das oclusões, da densidade da cena e da avaliação escolhida. Vamos detalhar a decisão específica que torna o ByteTrack útil nesta prática.

## 08. ByteTrack e detecções de menor confiança

**Imagem:** [08-bytetrack.png](assets/infograficos/08-bytetrack.png)

**Objetivo:** explicar a associação em duas etapas e o cuidado com o filtro anterior ao tracker.

**Indicação de apresentação:** acompanhar primeiro o candidato de alta confiança e depois a trajetória recuperada pela detecção fraca. Pausa de quatro segundos.

**Fala**

Imagine um veículo parcialmente encoberto. O detector ainda produz uma caixa, mas atribui a ela confiança menor. Se descartarmos essa caixa imediatamente, o rastreador perde uma evidência que poderia manter a trajetória.

O ByteTrack organiza a associação em duas etapas. Primeiro, tenta associar as detecções de maior confiança. Depois, usa candidatos de menor confiança para tentar recuperar trajetórias que permaneceram sem correspondência.

A interpretação é importante: uma detecção fraca pode ser compatível com algo que já está sendo acompanhado. Isso não significa que qualquer caixa fraca deva iniciar uma nova identidade. Na segunda etapa, a evidência é usada para recuperação, sob os critérios de compatibilidade do rastreador.

No código, precisamos separar o limiar mínimo do detector, a divisão entre as faixas de confiança e o critério de ativação de uma trajetória. Se filtrarmos antecipadamente toda a faixa fraca, a segunda etapa ficará sem esses candidatos.

Mesmo um quadro sem detecções deve atualizar o estado temporal. Com essa lógica definida, resta garantir que todas as operações usem coordenadas e convenções geométricas consistentes.

## 09. Coordenadas, transformações e âncoras

**Imagem:** [09-coordenadas.png](assets/infograficos/09-coordenadas.png)

**Objetivo:** conectar a representação das caixas à semântica das regras espaciais.

**Indicação de apresentação:** comparar a mesma caixa nos três formatos e, depois, apontar o ponto escolhido como âncora. Pausa de quatro segundos.

**Fala**

Uma mesma caixa pode ser representada por dois cantos, por um canto e suas dimensões, ou pelo centro e suas dimensões. Trocar o nome do formato sem transformar os valores altera a geometria interpretada pelo programa.

Na convenção raster dos notebooks, a origem fica no canto superior esquerdo. A coordenada horizontal cresce para a direita, e a vertical cresce para baixo. Já o acesso ao array usa primeiro a linha e depois a coluna: imagem de y, x.

Ao normalizar, dividimos coordenadas horizontais pela largura e verticais pela altura. Isso não transforma pixels em metros. Se houver redimensionamento e preenchimento de bordas, precisamos considerar tanto a escala quanto o deslocamento introduzido. Alguns adaptadores já devolvem caixas na imagem original; aplicar a correção novamente produz um erro.

A âncora define qual parte da caixa participa de uma regra. No projeto final, contamos a passagem completa dos quatro cantos. Enquanto a caixa intercepta a linha, ela não estabelece um lado inequívoco. O centro continua útil para desenhar rastros.

A LineZone também restringe a observação à faixa delimitada pelas perpendiculares que passam pelos extremos do segmento. Portanto, a reta não funciona como um limite infinito de contagem. Essas decisões geométricas precisam acompanhar a interpretação dos resultados.

## 10. Desafios e interpretação das saídas

**Imagem:** [10-desafios.png](assets/infograficos/10-desafios.png)

**Objetivo:** delimitar conclusões e preparar a passagem para os notebooks.

**Indicação de apresentação:** apontar oclusão, escala, ambiguidade e movimento da câmera; terminar retomando o vídeo completo. Pausa final de quatro segundos.

**Fala**

Em imagens aéreas, veículos pequenos podem ocupar poucos pixels. Oclusões removem partes da evidência visual. Objetos semelhantes competem por associações, e o movimento da câmera desloca simultaneamente várias regiões da cena.

Uma linha fixa em pixels pode deixar de representar a mesma seção da rua quando a câmera se move. Da mesma forma, um rastro na imagem não mede automaticamente deslocamento no solo. Seriam necessárias hipóteses e transformações adicionais para interpretar distâncias ou velocidades físicas.

Na avaliação, precisamos separar execução correta do código, consistência das coordenadas, plausibilidade das associações e desempenho medido contra uma referência. Uma visualização legível não demonstra que todas as identidades estejam corretas. Totais iguais também podem esconder omissões e duplicações que se compensam.

A sequência prática seguirá essas distinções. Primeiro, usaremos Supervision para organizar detecções, converter representações e construir anotações. Depois, acompanharemos objetos ao longo de um vídeo. Finalmente, transformaremos trajetórias em eventos e confrontaremos esses eventos com uma referência manual independente.

Ao analisar o resultado, mantenha três perguntas separadas: quantos objetos aparecem agora, quantos identificadores foram emitidos e quantas passagens foram observadas. Cada resposta descreve uma grandeza diferente.

## Referências de apoio, fora da narração

A fundamentação geral e as versões da implementação estão em [FUNDAMENTOS.md](FUNDAMENTOS.md) e [REFERENCIAS.md](REFERENCIAS.md). Fontes adicionais conferidas em 26/09/2026:

- **Imagem 02:** [Foundations of Computer Vision: Image Derivatives](https://visionbook.mit.edu/derivatives.html). A narração declara a convenção raster dos notebooks; a orientação deve usar `atan2(g_y, g_x)` nesse mesmo sistema.
- **Imagem 04, SIFT:** [OpenCV: Introduction to SIFT](https://docs.opencv.org/4.13.0/da/df5/tutorial_py_sift_intro.html), em conjunto com o artigo de Lowe já listado nas referências.
- **Imagem 04, SURF:** [OpenCV: Introduction to SURF](https://docs.opencv.org/4.13.0/df/dd2/tutorial_py_surf_intro.html) e [API SURF](https://docs.opencv.org/4.13.0/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html). A assinatura da API explicita `extended=False`, correspondente a 64 componentes; `extended=True` seleciona 128.
- **Imagem 04, ORB:** [OpenCV: ORB](https://docs.opencv.org/4.13.0/d1/d89/tutorial_py_orb.html) e [API ORB](https://docs.opencv.org/4.13.0/db/d95/classcv_1_1ORB.html). A configuração narrada usa `WTA_K=2`, 32 bytes, equivalentes a 256 bits, e distância de Hamming. Com `WTA_K` igual a 3 ou 4, a implementação requer a variante `NORM_HAMMING2`.
- **Imagem 09:** [API LineZone](https://supervision.roboflow.com/latest/detection/tools/line_zone/), para âncoras e limites da faixa de observação.
