# Configurar a GPU no RunPod

Este guia prepara uma máquina com GPU para executar os notebooks **01 e 02** do Projeto 1.

Crie sua conta no [RunPod](https://runpod.io/).
Antes de contratar recursos, consulte os preços no painel. O treinamento de referência levou
cerca de 2h30 em uma RTX 4090, mais preparação e avaliação. GPU ociosa e armazenamento também
podem gerar cobrança.

## Configuração passo a passo

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
