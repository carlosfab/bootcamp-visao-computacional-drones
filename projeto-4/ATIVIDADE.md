# Projeto 4 · Quantos bovinos aparecem nesta imagem?

O projeto usa o recorte didático `gado-icaerus-v1`, derivado do ICAERUS v2.
São **300 imagens: 180 de treino, 60 de validação e 60 de teste**, com uma classe,
`cow`. Os pacotes, as partições e os hashes acompanham o repositório.

O quinto notebook acrescenta uma investigação de treinamento com recortes e um
desafio de **62 capturas reservadas**, todas positivas. Essa extensão foi motivada
pelos resultados já observados em Derval; não transforma sua reavaliação em um
novo teste independente.

## Situação

Uma equipe deseja contar e localizar bovinos em fotografias de drone. Você deverá
comparar um detector genérico, o mesmo detector ajustado ao conjunto e uma versão
com inferência em recortes sobrepostos. Ao final, recomende uma configuração com
base nos erros encontrados e no custo do processamento.

Seu resultado representa **bovinos anotados por imagem**. Ele não é um censo de
animais únicos da fazenda: imagens diferentes podem mostrar os mesmos indivíduos.

## Objetivo

Produzir um experimento que outra pessoa consiga repetir, documentando os dados,
as configurações e a avaliação. Não há uma pontuação mínima de F1 ou uma melhoria
obrigatória para concluir o projeto. A qualidade da entrega depende da validade
da comparação e da interpretação das evidências.

## Percurso

Execute os notebooks na ordem abaixo. Cada arquivo também pode ser executado de
forma independente; o notebook `03` inclui um checkpoint de referência para a comparação
entre imagem inteira e recortes. Preserve as partições fornecidas e salve os
resultados antes de encerrar o runtime.

- `00_dados_e_problema.ipynb`: auditoria, anotações e escala; CPU suficiente.
- `01_baseline_e_contagem.ipynb`: YOLO11n genérico, calibração e teste.
- `02_fine_tuning.ipynb`: treinamento de 25 épocas, calibração e teste.
- `03_objetos_pequenos.ipynb`: mesmo checkpoint, imagem inteira e recortes de 640 pixels.
- `04_treino_com_recortes.ipynb`: treino de 20 épocas em recortes; calibração nas
  fotografias originais de validação; reavaliação exploratória de Derval e
  comparação com COCO em 62 capturas reservadas.

Recomenda-se GPU para inferência e treinamento. As bibliotecas têm versões
fixadas em `requirements.txt`, e cada notebook registra a versão do PyTorch e do
Python do ambiente. Não é necessária conta no Roboflow, chave de API ou acesso a
GPU pago para obter os dados. O tempo de execução depende do hardware disponível.

Nos quatro primeiros notebooks há **dois experimentos pareados pelas mesmas imagens de teste**:

- Baseline versus fine-tuning: mesma arquitetura e mesmo caminho de inferência,
  com pesos iniciais e pesos ajustados distintos.
- Imagem inteira versus recortes: exatamente o mesmo checkpoint de referência
  nos dois métodos do notebook `03`.

O checkpoint de referência pode diferir do seu treino no notebook `02`. O
relatório deve identificar o hash dos pesos em cada linha e manter os dois pares
explícitos. Não atribua diferenças entre checkpoints exclusivamente ao tiling.
Para estudar recortes no seu próprio peso, avalie novamente **inteira e recortes**
com esse peso, calibrando ambos na validação antes da avaliação congelada.

O quinto notebook introduz um terceiro comparativo: modelo treinado em recortes
versus COCO, **ambos com inferência em recortes**. Cada limiar é calibrado
separadamente nas mesmas 60 fotografias originais de validação. A configuração é
congelada antes de abrir a reserva; não escolha parâmetros por seus resultados.

As 62 capturas vêm da pasta agregada `Other_farms`: o nome não garante uma única
fazenda nem independência geográfica. Todas são positivas, por isso o comparativo
não estima a taxa de falsos alarmes em cenas vazias. A reavaliação das 60 imagens
de Derval nesse quinto notebook deve aparecer separadamente como **exploratória**.

| Etapa | O que investigar | Evidência esperada |
|---|---|---|
| Dados | Origem, licença, imagens, anotações e separação entre partições | Resumo da auditoria e exemplos anotados |
| Baseline | Detecção genérica de bovinos vistos de cima | Previsões, métricas e casos de erro |
| Fine-tuning | Efeito do ajuste ao conjunto | Configuração, checkpoint selecionado e comparação controlada |
| Recortes | Efeito da escala de entrada e da deduplicação | Comparação com imagem inteira e inspeção da sobreposição |
| Treino com recortes | Compatibilidade de escala e nova avaliação após uma hipótese | Comparativo reservado com COCO e limites da reserva |
| Decisão | Benefício, custo e limites de cada configuração | Recomendação apoiada em tabela e exemplos |

## Perguntas sobre os dados

Registre quantas imagens existem em cada partição, usando o manifesto. Explore
caixas, dimensões e tamanho aparente somente no treino e na validação durante o
desenvolvimento. Mostre pelo menos um caso em que a resolução possa dificultar a
detecção. As estatísticas de caixas do teste entram no relatório após a avaliação
com as configurações congeladas.

Treino e validação usam Mauron e Jalogny em datas separadas. No protocolo inicial
dos notebooks 00–03, Derval era a fazenda reservada para teste. Seus resultados
motivaram a hipótese do notebook 04; nessa etapa, Derval é exploratória e as
62 novas capturas constituem a reserva. Confira os grupos no manifesto.
Não há identificação individual dos bovinos, portanto datas diferentes podem
mostrar os mesmos animais. As imagens foram reduzidas proporcionalmente para
lado maior de até 2048 pixels e recomprimidas em JPEG; essa adaptação limita o
detalhe disponível mesmo antes de alimentar a rede.

Não confunda ausência de arquivos idênticos com ausência de cenas próximas.
Recortes da mesma imagem devem permanecer na mesma partição. Nunca use imagens
do teste para gerar recortes de treino. Se os resultados do teste orientarem uma
nova hipótese, esse conjunto passa a ser exploratório: reserve novas capturas
antes da avaliação final, como foi feito no notebook 04.

## Regras da comparação

- Use o treino para ajustar os pesos e a validação para escolher checkpoint e parâmetros.
- Registre como foi escolhido o limiar de confiança de cada método.
- Fixe configurações antes de avaliar o teste.
- Compare os métodos sobre as mesmas imagens de teste, sem retirar casos difíceis.
- Confira o mapeamento da classe de bovino entre o detector genérico e o conjunto.
- Use a mesma regra documentada de associação entre caixas em todas as configurações.
- Ao comparar imagem inteira e recortes, mantenha o mesmo checkpoint ajustado.
- Identifique o SHA-256 dos pesos e o par experimental de cada resultado.
- No quinto notebook, recorte somente treino e validação para o ajuste; não
  acrescente Derval ou a reserva ao treinamento.
- Separe o resultado exploratório em Derval do desafio de 62 capturas reservadas.
- Declare qualquer alteração adicional que impeça atribuir o efeito a uma única mudança.

Se um experimento posterior for motivado por um resultado do teste, identifique-o
como exploração. Ele não constitui outra avaliação final independente sobre os
mesmos dados.

No quinto notebook, a nova estratégia também muda épocas, batch, mosaico e
variação de escala. Uma eventual melhora não isola causalmente o efeito do
recorte: a comparação avalia esse conjunto de decisões.

## Avalie detecção e contagem

Apresente precisão, recall e F1 com associação um a um e IoU de pelo menos 0,5,
no limiar escolhido na validação. Essas métricas não são AP ou mAP. A regra de
calibração é maior F1; em empate, menor MAE e depois maior limiar. Inclua o número
de imagens avaliadas, erro absoluto
médio de contagem, viés e tempo de processamento por imagem.

Para cada imagem, a contagem de referência é o número de caixas anotadas; a
contagem prevista é o número de caixas após filtragem e deduplicação. Uma previsão
com a contagem correta pode conter falsos positivos e falsos negativos.

Se usar as fórmulas abaixo, considere $N$ imagens, $y^{(i)}$ bovinos anotados e
$\hat{y}^{(i)}$ caixas previstas na imagem $i$:

$$
\operatorname{MAE} = \frac{1}{N}\sum_{i=1}^{N}
\left|\hat{y}^{(i)}-y^{(i)}\right|.
$$

$$
\operatorname{Viés} = \frac{1}{N}\sum_{i=1}^{N}
\left(\hat{y}^{(i)}-y^{(i)}\right).
$$

Viés negativo indica tendência de subcontagem; positivo, de supercontagem.
Viés próximo de zero pode resultar da compensação entre erros. O MAE tem unidade
de bovinos por imagem e não é uma porcentagem.

Separe também imagens positivas e negativas. A taxa de contagem exata agregada
pode ser alta porque muitas cenas não contêm bovinos anotados. Nos negativos,
informe a fração de imagens que receberam ao menos um falso positivo.

Na reserva do quinto notebook não há negativos. Declare a taxa de falso alarme
em cenas vazias como **não estimável**, sem convertê-la em zero. Falsos positivos
de caixas ainda podem ocorrer e ser medidos nas cenas positivas.

Na medição de tempo, informe hardware, quantidade de imagens e o que entrou na
medida. Para recortes, registre também quantos foram processados. Use o mesmo
escopo de medição entre os métodos. Se não medir uma etapa, indique que ela não
foi medida.

## Investigue as falhas

Escolha pelo menos quatro imagens com comportamentos relevantes e identifique
em qual partição foram observadas. Inclua exemplos de erro; não apresente somente
as melhores previsões. Quando presentes, procure omissões de animais pequenos,
confusão com o fundo, bovinos próximos e duplicação na sobreposição de recortes.

Para cada exemplo, mostre referência e previsão e explique:

- Qual foi o erro observado?
- Qual etapa pode explicá-lo: dados, detector, limiar, resolução ou deduplicação?
- Que evidência apoia sua explicação e o que ainda precisaria ser verificado?

Se uma hipótese de falha não ocorrer nas imagens avaliadas, registre isso. Não é
necessário fabricar um exemplo nem supor que toda categoria de erro estará presente.

## Entrega

Entregue uma pasta ou arquivo compactado contendo:

- Notebooks executados, com saídas e sem erros de execução ocultos.
- Manifesto dos dados utilizados, com versões, partições e identificadores dos arquivos.
- Registro do ambiente e das configurações de treino, inferência e avaliação.
- Checkpoint ajustado ou referência estável para obtê-lo, acompanhado de seu hash.
- Previsões e contagens por imagem, identificadas por configuração.
- Tabelas comparativas com hash dos pesos, par experimental e conjunto avaliado em cada linha, e imagens
  anotadas usadas na análise de falhas.
- Relatório curto com recomendação final e limitações do experimento.

O relatório deve responder: qual configuração você recomenda para as imagens
avaliadas; por que ela é adequada; qual é o custo observado; quais falhas exigem
revisão humana; o que falta para estudar uma fazenda ou um voo ainda não observado.

Preserve licença e atribuições na redistribuição de dados, imagens ou pesos.
Não inclua chaves de acesso ou credenciais. Resultados de referência devem ser
identificados como tal e distinguidos daqueles recalculados na sua execução.

As pastas de métricas e figuras são atualizadas ao reexecutar. O nome de uma
pasta de treino existente é protegido, mas essa proteção não se estende a todos
os relatórios. Antes de repetir, use as instruções de compactação e download ao
final de cada notebook ou no README. Salve separadamente o notebook com saídas.

## Critérios de avaliação

| Critério | Pontos | O que caracteriza uma entrega completa |
|---|---:|---|
| Auditoria e entendimento dos dados | 15 | Contagens verificadas, exemplos, escala dos animais, origem e limites dos rótulos |
| Separação e protocolo experimental | 20 | Partições preservadas, risco de vazamento discutido e decisões tomadas sem usar o teste |
| Comparação entre baseline e fine-tuning | 15 | Classe mapeada corretamente, configuração registrada e efeito do ajuste interpretado |
| Recortes e deduplicação | 15 | Coordenadas consistentes, custo/benefício, treino recortado sem teste e comparação reservada |
| Avaliação e análise dos erros | 20 | Métricas de detecção e contagem, falhas concretas e conclusões compatíveis com os resultados |
| Reprodução e recomendação final | 15 | Artefatos completos, ambiente identificável, percurso executável e decisão fundamentada |
| **Total** | **100** | |

O professor atribuirá pontos parciais conforme as evidências entregues. Uma
configuração que não melhore o baseline pode receber a pontuação completa quando
o experimento estiver correto e a interpretação for fundamentada. Uma métrica
alta, isoladamente, não compensa vazamento, comparação inconsistente ou resultado
sem origem verificável.

## Extensões opcionais

Depois de concluir o núcleo, escolha uma investigação adicional na validação:
variar o tamanho do recorte, examinar o efeito da sobreposição ou comparar erros
por faixa de tamanho aparente. Registre a pergunta antes de executar e mantenha
as demais variáveis controladas.

Um estudo posterior pode reservar outros voos e fazendas. O teste em Derval é
um caso de transferência para outra fazenda, sem demonstrar generalização para
qualquer propriedade ou condição de coleta.

Depois da hipótese do quinto notebook, Derval faz parte da análise exploratória.
A reserva adiciona capturas não usadas na escolha daquela configuração, mas não
substitui um estudo com separação geográfica verificável e cenas negativas.
