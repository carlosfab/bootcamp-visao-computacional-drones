# Projeto 4 · Detecção e contagem de gado em imagens de drone

**Guia de condução do professor.** O recorte `gado-icaerus-v1` usa ICAERUS v2:
300 imagens, divididas em 180 de treino, 60 de validação e 60 de teste. Os
notebooks foram escritos para execução independente e incluem dados versionados.
Antes da aula, conferir os registros de execução e inspecionar os resultados
produzidos no ambiente escolhido. Este guia não atribui métricas ou tempos a uma
execução que ainda não tenha sido realizada.

## O problema que organiza a aula

Uma equipe quer estimar quantos bovinos aparecem em fotografias aéreas e localizar
os animais para conferir a contagem. O desafio é decidir se um detector genérico
serve, se o treinamento com imagens do domínio melhora o resultado e se processar
recortes recupera animais pequenos sem produzir duplicatas demais.

A unidade de avaliação é **uma imagem e suas anotações**. Essa escolha permite
examinar cada omissão e cada falso positivo. Fotografias sobrepostas não permitem,
por simples soma, concluir quantos animais diferentes existem no rebanho.

O produto final será uma recomendação fundamentada entre três configurações:
detector genérico, detector ajustado ao conjunto e detector ajustado com inferência
em recortes. O aluno deve conseguir recomendar manter uma configuração mais simples
se a alternativa não oferecer benefício suficiente.

## O que o aluno já sabe e o que acrescentamos

O Projeto 1 estabeleceu o percurso de preparação, treinamento, validação e teste.
O Projeto 2 discutiu mudança de domínio e tamanho aparente dos objetos. O Projeto 3
ensinou representações, associações temporais e contagem de passagens.

Aqui, retome essas ideias em função de uma pergunta concreta: **a contagem produzida
é boa o suficiente para o uso delimitado?** A novidade está em relacionar a decisão
de resolução, os erros de localização, o erro de contagem e o custo do processamento.
Não é necessário acrescentar outro rastreador.

Ao final, o aluno deve conseguir:

- Auditar imagens, anotações e partições antes de treinar.
- Diferenciar bovinos anotados, caixas previstas e animais únicos no rebanho.
- Comparar modelos sobre as mesmas imagens, com uma regra de avaliação explícita.
- Explicar por que a contagem pode estar correta mesmo quando a detecção está errada.
- Projetar caixas de recortes para a imagem original e explicar a deduplicação.
- Registrar configurações e limitações suficientes para outra pessoa repetir o experimento.

## Organização sugerida

Os tempos abaixo são **estimativas de planejamento pedagógico**, não medições de
execução dos notebooks. O download e o treinamento dependem da rede, da GPU e do
tamanho do conjunto. Ajuste a divisão após o ensaio completo.

| Encontro | Foco | Duração sugerida de condução |
|---|---|---|
| Primeiro | Problema, auditoria, baseline e significado dos erros | 90 minutos |
| Segundo | Fine-tuning, resolução, recortes e decisão final | 90 minutos |
| Trabalho independente | Executar, investigar falhas e organizar a entrega | 2 a 4 horas, além de eventuais esperas de processamento |

Prepare uma execução completa antes da aula. Ela permite discutir resultados caso
um runtime seja interrompido, mas deve aparecer como **execução de referência**, com
seus arquivos e configuração identificados. Não apresentar resultados salvos como
se tivessem sido calculados na sessão do aluno.

## Abertura: qual número estamos tentando produzir?

Mostre uma fotografia sem previsões. Peça à turma que estime quantos bovinos
consegue ver e que aponte regiões incertas: sombra, oclusão, borda da imagem ou
animais muito próximos. Em seguida, mostre as anotações do conjunto.

Fala de apoio:

> Antes de carregar o modelo, precisamos combinar o que significa acertar.
> Estamos estimando quantos bovinos foram anotados nesta imagem. Ainda não estamos
> medindo todos os animais de uma fazenda, nem identificando cada indivíduo.

Explore a diferença entre uma referência anotada e uma verdade infalível. Se uma
anotação parecer ausente ou ambígua, registre o caso. Não altere silenciosamente o
teste para favorecer o detector. Uma correção de rótulos exige uma nova versão dos
dados e a reavaliação de todas as configurações comparadas.

Verificação de compreensão: duas fotos com dez caixas em cada uma não demonstram
que há vinte animais diferentes. Elas podem mostrar os mesmos bovinos.

## Dados: a avaliação começa antes da rede

No notebook de dados, conduza a exploração nesta ordem: origem e condições de uso;
quantidade de imagens; caixas de treino e validação; exemplos anotados; dimensões
das imagens; tamanho das caixas; composição das partições. Só depois apresente o
carregamento pelo modelo. As caixas e imagens de teste permanecem fora da análise
exploratória, além da verificação automática de integridade dos arquivos.

Fala de apoio:

> Se o teste contém uma fotografia quase igual à usada no treinamento, o modelo
> pode parecer preparado para situações que ele ainda não enfrentou. Um arquivo
> diferente não garante uma cena diferente.

O recorte usa Mauron e Jalogny para treino, com datas anteriores a setembro de
2023, e validação posterior nessas fazendas. O teste usa exclusivamente Derval.
Os voos e os pares fazenda/data não atravessam partições. Um recorte herda a
partição de sua imagem original; separar depois de gerar recortes criaria uma
fonte adicional de vazamento. Não há identificação individual dos bovinos.

Hashes detectam cópias exatas. Comparações visuais ou de similaridade ajudam a
encontrar quadros próximos. Nenhuma dessas verificações, isoladamente, comprova
independência entre fazendas. Registre o que foi possível conferir e o que os
metadados não permitem concluir.

As imagens derivadas têm lado maior limitado a 2048 pixels, com redimensionamento
proporcional e recompressão JPEG. Os rótulos YOLO normalizados foram preservados.
O treino foi enriquecido com positivos; validação e teste procuram preservar a
proporção da fonte. Discuta como imagens vazias podem mascarar falhas em positivos
quando se reporta apenas a contagem exata agregada.

Conecte dimensão da imagem com dimensão do animal. Reduzir uma foto grande até a
entrada do detector também reduz cada bovino. Não atribua diferença de tamanho à
altitude quando o conjunto não fornecer esse metadado.

## Baseline: uma previsão é uma hipótese

Carregue um detector genérico e selecione a classe de bovino pelo nome. Confira o
mapeamento dessa classe para o rótulo único do conjunto antes de calcular métricas.
Uma classe com índice incorreto pode invalidar a comparação mesmo quando o desenho
das caixas parece plausível.

Comece por uma imagem de treino ou validação. Mostre as previsões e só então
compare com as caixas de referência. Discuta animais omitidos, caixas em objetos
errados, duplicatas e caixas deslocadas.

Fala de apoio:

> O modelo pode prever oito caixas para uma imagem com oito bovinos anotados e
> ainda assim errar. Uma caixa falsa e um animal esquecido se cancelam na soma.
> Por isso precisamos avaliar tanto a contagem quanto a correspondência entre caixas.

Use associação um a um entre previsões e anotações, com IoU de pelo menos 0,5.
Apresente precisão, recall e F1 no ponto de operação escolhido. O avaliador do
projeto não calcula AP ou mAP. Na contagem, examine erro absoluto por
imagem e viés: o sistema tende a subcontar ou a supercontar?

O limiar de confiança é escolhido pelo maior F1 na validação; em empate, menor
MAE e depois maior limiar. Ao
comparar limiares, mostre também o efeito em FP e FN. Não use o resultado de uma
imagem do teste para decidir o valor.

## Fine-tuning: mudar uma coisa que conseguimos interpretar

Use YOLO11n tanto no baseline quanto no ajuste. O notebook fixa 25 épocas,
batch 8, entrada 640, semente 42, dois workers, `deterministic=True`, sem cache
nem geração automática de gráficos. O PyTorch do runtime é preservado. Os
argumentos completos ficam em `args.yaml`, e `best.pt` é selecionado pela
validação durante o treino. Explique qual parte do experimento está sendo alterada.

Fala de apoio:

> Agora mostramos ao modelo como os bovinos aparecem neste conjunto de imagens.
> A pergunta é se esse ajuste melhora o resultado em imagens que não foram usadas
> para atualizar os pesos. O treinamento terminar sem erro não responde a isso.

Durante a espera do treino, retome exemplos da validação e discuta hipóteses de
erro. Preserve o teste. Não é necessário acompanhar cada atualização do treinamento
para entender a seleção do melhor checkpoint.

Separe a comparação de pesos da comparação de resolução: ao contrastar baseline
e fine-tuning, use o mesmo percurso de inferência. Caso seja necessário alterar
mais de uma variável, informe a mudança e evite atribuir o efeito apenas ao treinamento.

Não prometa que um treinamento curto superará o modelo genérico. Um resultado
desfavorável pode revelar problema nas anotações, amostra pequena, configuração
inadequada ou domínio insuficientemente representado. Cabe ao aluno apoiar sua
interpretação em evidências.

## Objetos pequenos: preservar detalhe custa processamento

Mostre a imagem original e a entrada redimensionada. Escolha na validação um
bovino pequeno e acompanhe quantos pixels ele ocupa em cada representação. Uma
ampliação da exibição não recupera a informação eliminada pelo redimensionamento.

Depois apresente recortes sobrepostos e a volta das caixas ao referencial original.
Faça explícitos o deslocamento do recorte e qualquer escala aplicada. O resultado
da contagem só é calculado após juntar e deduplicar as previsões.

Fala de apoio:

> Cada recorte permite que o animal ocupe uma parte maior da entrada da rede.
> Mas o mesmo animal pode aparecer em dois recortes. Ganhar detalhe cria também
> um problema de duplicação que precisamos resolver.

O notebook fixa recortes de 640 pixels, sobreposição nominal de 20% e NMS global
com IoU 0,5. O checkpoint de referência acompanha o projeto, permitindo executar
esta comparação sem repetir o treino. Seu hash aparece no registro da avaliação.

Mostre uma região de sobreposição antes e depois da deduplicação. Discuta dois
tipos de erro: manter duas caixas para um bovino ou remover uma caixa de um segundo
bovino próximo. Recortar também pode reduzir o contexto disponível ao detector.

A comparação principal usa o mesmo checkpoint ajustado. Tamanho dos recortes,
sobreposição, regra de deduplicação e limiar são definidos na validação. Registre
quantos recortes foram processados e como o tempo foi medido, com o mesmo escopo
de medição entre os métodos. Um tempo de inferência não representa automaticamente
o tempo de download, preparação e produção dos arquivos.

Como o conjunto contém imagens previamente reduzidas, trate
o tiling como investigação. Não apresente ganho de detalhe inexistente nem suponha
que o método será superior.

## Avaliação final e decisão

Antes da avaliação final, registre as configurações que serão comparadas. Congele
checkpoint, limiar, tamanho de entrada, recortes e deduplicação. A tabela final deve
usar as mesmas imagens de teste e indicar o número de imagens efetivamente avaliadas.

Uma comparação planejada de configurações fixadas pode usar o mesmo teste. Escolher
novos parâmetros depois de ler seus resultados transforma esse teste em parte do
desenvolvimento; nesse caso, o aluno deve declarar a mudança e reconhecer a
necessidade de outro conjunto para uma avaliação final independente.

Peça uma decisão em linguagem aplicada:

> Qual configuração você recomenda para estas imagens? Que melhoria ela entrega,
> quanto custa processá-la e em quais situações você ainda exigiria conferência manual?

Uma resposta completa combina tabela quantitativa, exemplos de erros e limites do
desenho experimental. Aumentar F1 não garante menor erro de contagem; reduzir o erro
médio não garante que os casos mais difíceis tenham sido resolvidos.

## Fechamento: contagem na imagem e censo do rebanho

Retome o Projeto 3 para lembrar que identidade é uma hipótese construída ao longo
do tempo. Um ID de rastreamento pode mudar ou ser reaproveitado; ele não funciona
automaticamente como identificação permanente de um bovino.

Para estimar animais únicos em uma área, seria necessário definir cobertura,
sobreposição, momento da coleta e tratamento de movimento e oclusão, além de uma
referência independente. Esses elementos não são fornecidos apenas por caixas
corretas em fotografias isoladas.

Feche pedindo três frases: o que o experimento demonstrou; o que permaneceu
incerto; qual seria a próxima coleta ou medição para reduzir essa incerteza.

## Preparação e pendências antes da aula

| Item | Configuração definida ou conferência necessária |
|---|---|
| Conjunto e versão exatos | ICAERUS v2, CC BY 4.0; recorte didático gado-icaerus-v1 |
| Recorte distribuído | 300 imagens; pacotes locais verificados por SHA-256 |
| Imagens e anotações | 180/60/60; classe cow; lado maior até 2048; examinar ambiguidades em treino/val |
| Grupos e partições | Datas separadas em Mauron/Jalogny; Derval somente no teste; voos separados |
| Configuração dos modelos | YOLO11n; 25 épocas; batch 8; 640 pixels; seed 42; best.pt pela validação |
| Tiling | 640 pixels, overlap 0,2, NMS global 0,5; mesmo checkpoint nos dois métodos |
| Ambiente do aluno | Executar de início ao fim no ambiente declarado e inspecionar saídas |
| Tempos de execução | Medir e registrar separadamente das estimativas pedagógicas |
| Resultados de referência | Publicar apenas os produzidos pela execução verificada |

As referências, a licença e as atribuições devem acompanhar a distribuição dos
dados e o material final. A conclusão desta lista exige alinhamento com o enunciado
em [ATIVIDADE.md](ATIVIDADE.md) e com as saídas reais dos quatro notebooks.
