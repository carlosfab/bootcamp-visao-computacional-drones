# Prompt para configurar o RunPod com ajuda de um agente

Você não precisa saber operar servidores para fazer este projeto. Copie **todo o bloco abaixo**
e cole em um agente de codificação — Claude Code, Codex CLI, ChatGPT ou similar — com acesso ao
seu terminal. Ele vai conduzir a configuração, pedindo a você apenas o que só você pode fornecer
(criar a conta, gerar a chave da API, confirmar gastos).

Antes de colar, faça duas coisas:

1. **Crie sua conta** em https://runpod.io?ref=oke6mnm6 e adicione crédito. US$ 10 são
   suficientes com folga para este projeto.
2. **Gere uma API key** em *Settings → API Keys* no painel do RunPod. Guarde-a: você vai
   fornecê-la ao agente quando ele pedir.

> **Sobre custo:** uma RTX 4090 custa em torno de US$ 0,70/hora. O projeto completo leva
> cerca de 3 horas, ou aproximadamente **US$ 2,00 a US$ 2,50**. Você paga pelo tempo em que
> o pod está ligado — inclusive parado sem fazer nada. Desligá-lo ao terminar é sua
> responsabilidade, e o prompt abaixo instrui o agente a lembrá-lo disso.

---

## O prompt (copie a partir daqui)

```
Preciso da sua ajuda para configurar uma GPU na nuvem (RunPod) e rodar dois notebooks de
treinamento de um detector de objetos. Eu nunca usei RunPod nem SSH, então explique o que
você está fazendo em cada etapa, em linguagem simples, e me avise sempre que uma ação for
gerar custo.

CONTEXTO DO QUE VOU RODAR

- Dois notebooks Jupyter que treinam um modelo YOLOv8s em um dataset de imagens térmicas
  aéreas (dataset POP, ~1,2 GB, baixado automaticamente pelo primeiro notebook).
- O treinamento leva cerca de 2h30 em uma RTX 4090 e usa ~14,3 GB de VRAM.
- Preciso de uma GPU com pelo menos 16 GB de VRAM.

RESTRIÇÕES TÉCNICAS IMPORTANTES (não altere estas versões)

- ultralytics == 8.2.0
- numpy < 2
- torch < 2.6  (use torch 2.5.1)

  O motivo: a ultralytics 8.2.0 carrega checkpoints com torch.load() sem passar
  weights_only=False. A partir do torch 2.6 esse argumento mudou de padrão para True e o
  carregamento do modelo falha. Se a imagem do pod vier com torch mais novo, crie um
  ambiente virtual separado com torch 2.5.1+cu121 em vez de tentar contornar de outra forma.

O QUE PRECISO QUE VOCÊ FAÇA

1. CHAVE SSH
   Verifique se já existe uma chave SSH no meu computador (~/.ssh/id_ed25519.pub). Se não
   existir, gere uma com ssh-keygen -t ed25519. Depois me mostre a chave PÚBLICA e me explique
   onde colá-la no painel do RunPod (Settings -> SSH Public Keys). Nunca me peça a chave
   privada e nunca a copie para outro lugar.

2. API KEY
   Me peça a minha API key do RunPod e guarde-a em uma variável de ambiente ou em um arquivo
   local fora de qualquer repositório git. Não a escreva em nenhum arquivo que eu possa vir a
   publicar, e não a imprima no terminal depois de configurada.

3. NETWORK VOLUME
   Crie um volume de rede de 20 GB para guardar o dataset e os resultados. O ponto importante:
   o volume sobrevive quando eu desligo o pod, então não perco o download nem os modelos
   treinados. Me diga em que data center ele foi criado — o pod terá que ser criado na mesma
   região.

4. POD COM GPU
   Crie um pod com:
   - uma GPU de pelo menos 16 GB (RTX 4090 é a referência; se não houver disponível, sugira
     alternativas e me diga o preço antes de criar)
   - o volume do passo 3 montado em /workspace
   - a porta 22 (SSH) exposta
   - uma imagem com PyTorch e CUDA
   Antes de criar, me diga o custo por hora e espere minha confirmação.

5. CONEXÃO
   Conecte via SSH e confirme que a GPU aparece (nvidia-smi). Me mostre o comando de conexão
   para que eu consiga entrar sozinho depois.

6. AMBIENTE
   Instale as dependências respeitando as versões da seção de restrições. Confirme que
   funcionou carregando um modelo YOLO de teste — não apenas importando a biblioteca:

     from ultralytics import YOLO
     m = YOLO("yolov8s.pt")

   Se isso falhar com erro de "weights only", a versão do torch está errada.

7. NOTEBOOKS
   Suba os arquivos 01_preparacao_dados.ipynb e 02_experimento_completo.ipynb para
   /workspace e inicie um Jupyter Lab acessível pelo navegador, me passando o link e o token.
   Alternativa, se for mais simples: configure o acesso por túnel SSH e me explique como abrir.

8. AO FINAL
   Me explique como:
   - verificar quanto já gastei
   - DESLIGAR o pod quando eu terminar (e a diferença entre parar e terminar)
   - garantir que os resultados ficaram salvos no volume antes de desligar

REGRAS DE SEGURANÇA

- Nunca me peça senha de conta.
- Nunca grave a API key em um arquivo dentro de um repositório git.
- Antes de qualquer ação que gere cobrança, me avise e espere confirmação.
- Se algo falhar duas vezes seguidas, pare e me explique o que aconteceu em vez de continuar
  tentando variações.

Comece pelo passo 1 e me guie um passo de cada vez, confirmando comigo antes de avançar.
```

---

## Se preferir fazer manualmente

A sequência é a mesma, sem o agente:

1. **Chave SSH** — `ssh-keygen -t ed25519`, e cole o conteúdo de `~/.ssh/id_ed25519.pub` em
   *Settings → SSH Public Keys*.
2. **Volume** — *Storage → New Network Volume*, 20 GB. Anote a região.
3. **Pod** — *Pods → Deploy*, escolha uma RTX 4090 **na mesma região do volume**, monte o
   volume em `/workspace` e exponha a porta 22.
4. **Conectar** — o painel mostra o comando `ssh root@<ip> -p <porta>`.
5. **Ambiente** — dentro do pod:

   ```bash
   python3 -m venv /opt/pop-env
   /opt/pop-env/bin/pip install torch==2.5.1 torchvision==0.20.1 \
       --index-url https://download.pytorch.org/whl/cu121
   /opt/pop-env/bin/pip install "numpy<2" ultralytics==8.2.0
   ```

6. **Notebooks** — envie com `scp` e rode via Jupyter, ou execute pelo terminal.
7. **Desligar** — *Terminate* remove o pod e para a cobrança. O volume continua existindo (e
   é cobrado à parte, por volta de US$ 0,07/GB por mês).

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `Weights only load failed` ao carregar o modelo | torch ≥ 2.6 | instale torch 2.5.1 |
| `CUDA out of memory` durante o treino | GPU com menos de 16 GB | reduza `BATCH` para 8 e registre a mudança |
| O pod não enxerga o dataset | volume montado em caminho errado | confirme que está em `/workspace` |
| SSH recusa a conexão | chave pública não cadastrada, ou cadastrada após criar o pod | recrie o pod depois de cadastrar a chave |
| Não há RTX 4090 disponível | região sem estoque | tente outra GPU ≥ 16 GB na **mesma região do volume** |
