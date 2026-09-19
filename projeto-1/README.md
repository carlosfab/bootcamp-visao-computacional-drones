# Projeto 1 · Detecção de pessoas em imagens térmicas aéreas para busca e salvamento

Neste projeto você treina um detector de pessoas em imagens térmicas capturadas por drone e
responde a uma pergunta que não se resolve com acurácia: **este modelo serve para uma operação
real de busca e salvamento?**

A diferença importa. Em uma missão real, deixar de encontrar uma pessoa e disparar um alarme
falso não custam a mesma coisa. Um modelo com ótimo mAP pode ser inútil se as pessoas que ele
perde forem justamente as parcialmente ocultas pela vegetação — que são o caso difícil e,
não por acaso, o caso real.

Você vai reproduzir o protocolo experimental de um artigo publicado, comparar seus números com
os dos autores, e aprender por que a **ordem** das decisões em um experimento determina se o
resultado final significa alguma coisa.

## O que você vai construir

Um detector YOLOv8s treinado no dataset **POP** (*Partially Occluded Person*), de Song et al.
(2025): 8.768 imagens térmicas aéreas, 26.811 pessoas anotadas, capturadas a 30, 50 e 70 metros
de altitude em ambientes naturais.

Ao final você terá:

- um modelo treinado e um **protocolo congelado** que registra, com hash criptográfico, qual
  modelo foi avaliado e com que configuração;
- métricas em duas camadas — a **técnica** (mAP) e a **da missão** (recall e falsos positivos
  por imagem);
- uma análise de **onde** o modelo erra: por cena, por altitude de voo e por tamanho aparente
  da pessoa;
- uma comparação honesta com os valores publicados pelos autores.

## Os notebooks

Execute nesta ordem:

| Notebook | O que faz | GPU? | Tempo |
|---|---|---|---|
| [`aluno/01_preparacao_dados.ipynb`](aluno/01_preparacao_dados.ipynb) | baixa o POP, verifica a integridade, converte o formato, audita os splits e procura vazamento | não | ~10 min |
| [`aluno/02_experimento_completo.ipynb`](aluno/02_experimento_completo.ipynb) | treina, escolhe o modelo, define o ponto de operação e mede o desempenho final | **sim** | ~3 h |

O notebook 01 roda em qualquer máquina. Rode-o antes de alugar GPU: além de preparar os dados,
ele é a parte em que você inspeciona o dataset — e entender os dados antes de treinar é o hábito
que este projeto mais quer ensinar.

## Como conseguir uma GPU

O notebook 02 precisa de uma GPU com **pelo menos 16 GB de VRAM** (a configuração usa ~14,3 GB).
Se você não tem uma, há dois caminhos.

### Caminho A · RunPod (recomendado)

Aluguel de GPU por hora. Uma RTX 4090 sai por volta de **US$ 0,70/hora**, e o projeto completo
custa aproximadamente **US$ 2,00 a US$ 2,50**.

👉 **[Criar conta no RunPod](https://runpod.io?ref=oke6mnm6)**

> Esse é um link de indicação. Usá-lo não altera o seu custo e ajuda a manter este material.

**Nunca usou RunPod, SSH ou terminal remoto?** Não tem problema — não é pré-requisito deste
projeto. Preparei um prompt pronto para você colar em um agente (Claude Code, Codex, ChatGPT)
que conduz toda a configuração passo a passo, explicando cada etapa:

📋 **[aluno/SETUP-RUNPOD-PROMPT.md](aluno/SETUP-RUNPOD-PROMPT.md)**

Esse arquivo também traz o passo a passo manual, para quem preferir, e uma tabela dos erros mais
comuns com a solução de cada um.

**Atenção ao custo:** você paga pelo tempo em que o pod está ligado, mesmo parado sem fazer nada.
Desligue ao terminar. Os resultados ficam salvos no volume de rede, que sobrevive ao desligamento.

### Caminho B · Google Colab

Em preparação. O Colab gratuito não dá conta deste treinamento: são ~2h30 de GPU contínua, acima
do limite das sessões gratuitas, que também desconectam sem aviso. Um Colab pago é viável, mas
exige adaptações (montar o Drive, salvar checkpoints periodicamente, retomar após desconexão)
que ainda não estão prontas.

Enquanto isso, se você tem Colab Pro e quer tentar, o notebook 02 já retoma automaticamente de
`last.pt` se for reexecutado — mas você precisará garantir que os dados e os checkpoints estejam
no Drive, e não no disco efêmero da sessão.

## Requisitos técnicos

```
ultralytics == 8.2.0
numpy       <  2
torch       <  2.6   (use 2.5.1)
```

Essas versões não são recomendação, são requisito.

A ultralytics 8.2.0 carrega checkpoints com `torch.load()` sem passar `weights_only=False`. A
partir do torch 2.6 esse argumento passou a valer `True` por padrão, e o carregamento do
`yolov8s.pt` falha. É um detalhe de compatibilidade entre bibliotecas, sem qualquer efeito sobre
o experimento — mas é o erro que mais trava quem monta o ambiente sozinho.

Fixar a versão da ultralytics tem outra razão, essa científica: entre versões mudam defaults de
augmentation, de perda e de pós-processamento. Se você rodar com outra versão e obtiver números
diferentes, não saberá se a diferença veio do seu hardware, da sua semente ou de um default que
mudou sem aviso.

## O que observar enquanto executa

O projeto foi desenhado em torno de quatro ideias. Se ao final você levar só isso, já valeu:

1. **O split não pode ser aleatório.** As imagens vêm de voos contínuos: dois quadros seguidos
   são quase idênticos. Dividir aleatoriamente coloca imagens quase iguais no treino e no teste,
   e o resultado mede memorização, não capacidade de generalizar. O notebook 01 mostra a
   evidência disso medida em pixels.

2. **Escolher o modelo é uma decisão, e ela tem lugar certo.** O checkpoint é escolhido na
   validação. Se você escolhesse olhando o teste, o número reportado no teste deixaria de ser
   uma estimativa honesta — passaria a medir o quanto você explorou aquele conjunto.

3. **mAP não é requisito de missão.** O notebook mede duas coisas separadas: a qualidade técnica
   (mAP) e o que a operação exige (recall com um teto de alarmes falsos). São perguntas
   diferentes, e a segunda é a que decide se o sistema é utilizável.

4. **Um requisito que nunca falha não é um requisito.** O alvo de recall pode não ser atingido —
   e, quando isso acontece, o procedimento registra a falha em vez de relaxar o critério até
   caber. Esse é o comportamento correto.

## Referências

- Song, Y. et al. **An infrared dataset for partially occluded person detection in complex
  environment for search and rescue.** *Scientific Data* 12, 2025.
  https://doi.org/10.1038/s41597-025-04600-0

O dataset é baixado automaticamente pelo notebook 01, a partir do repositório oficial no OSF,
com verificação de integridade por SHA-256.
