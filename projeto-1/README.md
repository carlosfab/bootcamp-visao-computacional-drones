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

O notebook 01 roda em qualquer máquina. Rode-o no seu computador antes de alugar GPU: é a parte
em que você inspeciona o dataset, e entender os dados antes de treinar é o hábito que este
projeto mais quer ensinar.

Uma ressalva honesta: você vai precisar rodá-lo **de novo** na máquina com GPU, porque os dados
precisam estar no mesmo lugar que o treinamento. São cerca de 6 minutos lá. O que você ganha
rodando antes é entendimento, não tempo.

## Como conseguir uma GPU

O notebook 02 precisa de uma GPU com **pelo menos 16 GB de VRAM** (a configuração usa ~14,3 GB).

⚠️ **A geração da placa também importa.** Como este projeto exige `torch < 2.6` (veja
*Requisitos técnicos*), e essa versão do PyTorch só traz código compilado até a arquitetura
`sm_90`, placas **Blackwell — como a RTX 5090 — não funcionam**: o treinamento falha com
`CUDA error: no kernel image is available for execution on the device`. Escolha uma RTX 4090,
RTX 3090, A100, A6000, L40S ou equivalente. A RTX 4090 é a referência deste material.

Se você não tem uma GPU assim, há dois caminhos.

### Caminho A · RunPod (recomendado)

Aluguel de GPU por hora. Uma RTX 4090 sai por volta de **US$ 0,70/hora**, e o projeto completo
custa aproximadamente **US$ 2,00 a US$ 2,50**.

👉 **[Criar conta no RunPod](https://runpod.io?ref=oke6mnm6)**

> Esse é um link de indicação. Usá-lo não altera o seu custo e ajuda a manter este material.

**Nunca usou RunPod, SSH ou terminal remoto?** Não tem problema — não é pré-requisito deste
projeto. Preparei um prompt pronto para você colar em um agente (Claude Code, Codex, ChatGPT)
que conduz toda a configuração passo a passo, explicando cada etapa:

📋 **[aluno/GUIA-RUNPOD.md](aluno/GUIA-RUNPOD.md)**

Esse arquivo também traz o passo a passo manual, para quem preferir, e uma tabela dos erros mais
comuns com a solução de cada um.

**Atenção ao custo:** você paga pelo tempo em que o pod está ligado, mesmo parado sem fazer nada.
Desligue ao terminar.

**Atenção ao volume:** os resultados só sobrevivem ao desligamento se estiverem no volume de rede.
O diretório `/workspace` existe mesmo quando nenhum volume foi anexado — e, nesse caso, tudo é
apagado junto com a máquina. Os notebooks avisam quando detectam essa situação, mas confira ao
criar o pod que o volume está montado em `/workspace` e que ele fica na mesma região da GPU.

### Caminho B · Google Colab

Em preparação. O Colab gratuito não dá conta deste treinamento: são ~2h30 de GPU contínua, acima
do limite das sessões gratuitas, que também desconectam sem aviso. Um Colab pago é viável, mas
exige adaptações (montar o Drive, salvar checkpoints periodicamente, retomar após desconexão)
que ainda não estão prontas.

Enquanto isso, se você tem Colab Pro e quer tentar, o notebook 02 já retoma automaticamente de
`last.pt` se for reexecutado. Mas saiba o que terá de adaptar: no Colab a variável `RAIZ` aponta
para `/content/projeto-pop`, que é apagado ao fim da sessão. Não basta montar o Drive — é preciso
editar a célula de caminhos nos dois notebooks para que `RAIZ` fique dentro do Drive.

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

1. **O split não pode ser aleatório.** As imagens vêm de voos: o drone avança a ~1 m/s e a
   câmera registra uma imagem por segundo, então quadros seguidos cobrem quase o mesmo terreno.
   Dividir aleatoriamente coloca imagens fortemente redundantes no treino e no teste, e o
   resultado passa a medir memorização em vez de capacidade de generalizar. O notebook 01 mostra
   essa redundância medida em pixels.

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
