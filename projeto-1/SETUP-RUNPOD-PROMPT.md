# Configurar a GPU no RunPod

Este guia prepara a máquina para executar os **dois notebooks desta pasta**. Você pode seguir
com um agente que tenha acesso ao terminal ou usar o procedimento manual abaixo.

Crie sua conta no [RunPod](https://runpod.io?ref=oke6mnm6). Este é um link de indicação.
Antes de contratar recursos, consulte os preços no painel. O treinamento de referência levou
cerca de 2h30 em uma RTX 4090, mais preparação e avaliação. GPU ociosa e armazenamento também
podem gerar cobrança.

## Prompt para o agente

Abra o repositório clonado no agente e cole:

```text
Ajude-me a configurar o RunPod para executar o Projeto 1 deste repositório.
Leia projeto-1/README.md e projeto-1/AGENTS.md; explique cada etapa em linguagem simples.

1. Verifique meu ambiente e se já tenho uma chave SSH. Se precisar gerar uma, preserve
   as chaves existentes. Mostre somente a chave pública e indique onde cadastrá-la
   no painel do RunPod. Nunca copie a chave privada para o servidor.

2. Prefira me orientar pelo painel. Se precisar da API, ajude-me a configurar uma chave
   dedicada em um arquivo local fora do Git, com permissão 600. Eu digitarei o segredo
   diretamente no computador, sem colá-lo na conversa nem incluí-lo em comandos registrados.

3. Confira disponibilidade de GPU e volume de rede na mesma região ANTES de contratar.
   Sugira uma RTX 4090; também servem placas compatíveis como RTX 3090, A100, A6000 ou L40S.
   Preciso de pelo menos 16 GB de VRAM e compatibilidade com torch 2.5.1. Não use Blackwell
   (por exemplo, RTX 5090). Mostre o custo da GPU e do armazenamento e aguarde minha
   autorização antes de criar recursos pagos. Um volume de rede de 20 GB é o ponto de partida.

4. Monte o volume de rede em /workspace e configure SSH. Confirme no painel que o volume
   foi anexado; a pasta /workspace existir não garante que seja um volume de rede.

5. No pod, confirme a GPU com nvidia-smi e instale uv, se necessário. Clone este repositório
   em /workspace, na mesma branch que estou usando, e entre em projeto-1.
   Execute uv sync --locked nesta pasta. O projeto seleciona Python 3.12 e já configura
   torch 2.5.1 e torchvision 0.20.1 com CUDA 12.1. Preserve o uv.lock.
   Registre o kernel POP (torch 2.5.1) dentro de .venv usando --sys-prefix. Confirme as versões, CUDA e o carregamento de
   YOLO("yolov8s.pt") no mesmo Python que executará os notebooks.

6. Inicie Jupyter dentro de projeto-1, no ambiente virtual, com autenticação e acesso
   por túnel SSH. Mostre como abrir o navegador e qual kernel selecionar. Preserve o
   token em âmbito local, sem incluí-lo no repositório.

7. Ajude-me a executar 01_preparacao_dados.ipynb e depois 02_experimento_completo.ipynb.
   Preserve a sequência experimental. Não recalibre o limiar com base no teste nem altere
   versões para contornar um erro. Se precisar mudar parâmetros, explique a mudança e use
   outro NOME_EXECUCAO, sem apagar resultados.

8. Ao terminar, confira comigo os arquivos salvos no volume e ajude-me a fazer uma cópia
   dos resultados. Explique como encerrar o pod, conferir cobranças e revogar uma API key
   temporária. O volume continua sendo cobrado até sua exclusão; só o remova depois de eu
   confirmar o backup e autorizar a exclusão.

Comece verificando os pré-requisitos locais. Peça informação quando ela for necessária
para continuar e obtenha autorização antes de contratar recursos.
```

## Procedimento manual

1. **SSH:** cadastre sua chave pública em *Settings → SSH Public Keys*. Se não tiver uma chave,
   gere-a com `ssh-keygen -t ed25519`, sem sobrescrever uma existente.
2. **Armazenamento e GPU:** confira disponibilidade na mesma região. Crie o volume de rede,
   selecione uma GPU compatível e anexe o volume a `/workspace`. Exponha a porta SSH.
3. **Conexão:** copie do painel o comando `ssh root@<ip> -p <porta>` e execute `nvidia-smi` no pod.
4. **Ambiente:** dentro do pod, com [uv instalado](https://docs.astral.sh/uv/getting-started/installation/):

   ```bash
   cd /workspace
   git clone https://github.com/carlosfab/bootcamp-visao-computacional-drones.git
   cd bootcamp-visao-computacional-drones
   git switch main
   cd projeto-1
   uv sync --locked
   uv run --locked python -m ipykernel install --sys-prefix --name pop-env --display-name "POP (torch 2.5.1)"
   uv run --locked python -c 'import torch; from ultralytics import YOLO; print(torch.__version__, torch.cuda.is_available()); YOLO("yolov8s.pt")'
   uv run --locked jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --allow-root
   ```

   Mantenha a autenticação padrão do Jupyter. O processo mostra uma URL com token.

5. **Navegador:** em outro terminal do seu computador, abra o túnel:

   ```bash
   ssh -N -L 8888:127.0.0.1:8888 root@<ip> -p <porta>
   ```

   Abra a URL local mostrada pelo Jupyter, com o token, e selecione **POP (torch 2.5.1)**.
   Execute o notebook 01 no pod, mesmo que já tenha feito a preparação no computador.
6. **Resultados:** guarde seus notebooks executados e a pasta
   `/workspace/projeto-pop/runs/<nome-da-execucao>`. Verifique a cópia antes de encerrar recursos.
7. **Encerramento:** encerrar (*Terminate*) apaga o armazenamento ligado ao ciclo de vida do pod;
   o volume de rede permanece e continua sendo cobrado. Confira as opções disponíveis no painel
   e a [documentação de armazenamento](https://docs.runpod.io/pods/storage/types).

## Problemas comuns

| Sintoma | O que verificar |
|---|---|
| `Weights only load failed` | Confira torch 2.5.1 e o kernel selecionado; rode `uv sync --locked` e reinicie o kernel |
| `no kernel image is available` | Use GPU compatível com este PyTorch; RTX 5090/Blackwell não serve |
| `CUDA out of memory` | Libere a GPU ou reduza BATCH para 8; use outro NOME_EXECUCAO e recomece |
| A configuração diverge da execução salva | Restaure os parâmetros originais para retomar ou use outro nome para um novo experimento |
| Catálogo ou dados não encontrados | Execute o notebook 01 nessa máquina; confira POP_RAIZ e a pasta de onde abriu Jupyter |
| Notebook usa outro Python | Selecione POP (torch 2.5.1); confirme o ambiente onde o kernel foi registrado |
| SSH recusa conexão | Confira chave pública cadastrada, endereço e porta fornecidos no painel |
| Resultados desapareceram | Confira qual armazenamento foi anexado e sua política de persistência |

Instalação de referência: [PyTorch 2.5.1 e torchvision 0.20.1](https://pytorch.org/get-started/previous-versions/).
