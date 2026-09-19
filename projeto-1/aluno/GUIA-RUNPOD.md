# Configurar a GPU no RunPod

## Configuração passo a passo

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
   # Jupyter dentro do ambiente, senão o notebook roda no Python do sistema
   /opt/pop-env/bin/pip install jupyterlab ipykernel
   /opt/pop-env/bin/python -m ipykernel install --name pop-env \
       --display-name "POP (torch 2.5.1)"
   ```

6. **Notebooks** — envie com `scp` e rode via Jupyter, selecionando o kernel
   **POP (torch 2.5.1)** — não o Python padrão.
7. **Desligar** — *Terminate* remove o pod e para a cobrança. O volume continua existindo (e
   é cobrado à parte, por volta de US$ 0,07/GB por mês).

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `Weights only load failed` ao carregar o modelo | torch ≥ 2.6 | instale torch 2.5.1 |
| `CUDA out of memory` durante o treino | GPU com menos de 16 GB | reduza `BATCH` para 8, apague o diretório da execução e treine do zero — ao retomar de `last.pt` a Ultralytics reusa o batch antigo |
| Resultados sumiram ao desligar | volume não estava montado em `/workspace` | recrie o pod com o volume anexado, na mesma região dele |
| O pod não enxerga o dataset | volume montado em caminho errado | confirme que está em `/workspace` |
| SSH recusa a conexão | chave pública não cadastrada, ou cadastrada após criar o pod | recrie o pod depois de cadastrar a chave |
| Não há RTX 4090 disponível | região sem estoque | tente outra GPU ≥ 16 GB na **mesma região do volume** |
