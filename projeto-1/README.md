# Projeto 1 · Detecção de pessoas em imagens térmicas aéreas

Você vai preparar os dados, treinar um detector YOLOv8s e avaliar se ele atende a um requisito
experimental de busca e salvamento: **encontrar pelo menos 90% das pessoas, com no máximo um
falso positivo por imagem**. Um resultado de mAP, sozinho, não responde a essa pergunta.

Todo o material necessário começa nesta pasta. O experimento usa o **POP** (*Partially Occluded
Person*), de Song et al. (2025): 8.768 imagens térmicas aéreas e 26.811 pessoas anotadas.
O notebook 00 explora os dados originais, antes da conversão e do treinamento.

**Artigo de referência:** Song et al. (2025), [Scientific Data, 12, 300](https://doi.org/10.1038/s41597-025-04600-0).
**Dataset oficial:** [POP no OSF](https://osf.io/kmcva/overview).

## Comece aqui

1. Clone o repositório e entre em `projeto-1`.
2. Para começar pela exploração, prepare o ambiente leve descrito abaixo e coloque o POP em `dados/POP`.
3. Execute **00_analise_exploratoria.ipynb**. Depois, no ambiente completo do projeto, execute
   **01_preparacao_dados.ipynb** para converter e preparar os dados.
4. Prepare a máquina com GPU. Se ela for outra máquina, clone o repositório e execute o notebook
   01 nela também, para colocar os dados no mesmo ambiente do treinamento.
5. Execute **02_experimento_completo.ipynb**, da configuração à análise final.
6. Salve os resultados indicados ao final deste guia antes de encerrar a máquina.

| Arquivo | Para que serve |
|---|---|
| [00_analise_exploratoria.ipynb](00_analise_exploratoria.ipynb) | Apresentação acadêmica do dataset, conferência da Tabela 4, exemplos visuais e distribuições; CPU, sem PyTorch |
| [requirements-eda.txt](requirements-eda.txt) | Dependências leves exclusivas da análise exploratória |
| [01_preparacao_dados.ipynb](01_preparacao_dados.ipynb) | Download, conversão COCO → YOLO, auditoria de cenas, escala dos alvos e inspeção visual; cerca de 10 minutos, sem GPU |
| [02_experimento_completo.ipynb](02_experimento_completo.ipynb) | Treinamento, seleção do checkpoint na validação, calibração, congelamento e teste; projeção de 1h10–1h30 com parada antecipada em uma RTX 4090, podendo chegar a cerca de 3 horas |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) e [.python-version](.python-version) | Ambiente uv: dependências, versões resolvidas e Python padrão |
| [GUIA-RUNPOD.md](GUIA-RUNPOD.md) | Configuração da GPU na nuvem e preservação dos resultados |
| [tests/test_notebooks.py](tests/test_notebooks.py) | Verificações locais da lógica dos notebooks, sem treino ou download |

Os tempos dependem de download, hardware e armazenamento. Na execução de 19/09/2026,
o treinamento com parada antecipada levou aproximadamente 53 minutos em uma RTX 4090. A estimativa total inclui preparação e avaliação; não é uma
garantia de duração em outra máquina ou execução.

## Protocolo do artigo e adaptação didática

Song et al. (2025) reportam treinamento por **100 épocas**. Na preparação deste material,
também realizamos uma execução de referência com **100 épocas completas**. Para fins
didáticos, esta versão mantém **100 como limite máximo** e usa **`patience=10`**, permitindo
encerrar após 10 épocas consecutivas sem melhora no critério de validação da Ultralytics.
O objetivo é reduzir o tempo de execução e o custo de GPU da atividade.

Essa escolha é uma adaptação nossa e deve ser mencionada ao comparar os resultados com o
artigo. A parada acompanha `0,1 × mAP50 + 0,9 × mAP50–95` na validação; o teste não participa
da decisão. A época de encerramento e o resultado podem variar em uma nova execução.
Se houver melhorias contínuas, o treinamento poderá cumprir as 100 épocas.

Na execução didática de **19/09/2026**, a parada ocorreu na **época 35**, preservando o melhor
checkpoint da **época 25**. A comparação dos 355 tensores do `best.pt` confirmou pesos
numericamente idênticos aos do melhor checkpoint da nossa referência de 100 épocas. Esse
resultado foi observado nesta repetição com a configuração fixada; não garante igualdade
em outros treinamentos.

O notebook registra o limite, a paciência e as épocas efetivas em `treinamento_concluido.json`
e `protocolo_congelado.json`. Se o treino terminar cedo, poderá não executar a fase final
sem mosaico, programada para as últimas 10 épocas do limite de 100.
Para repetir o orçamento completo, use `PACIENCIA=0` nos notebooks 01 e 02 e um novo
`NOME_EXECUCAO`; zero desativa a parada antecipada. Preserve a execução anterior.

## Análise exploratória em CPU: notebook 00

O [notebook 00](00_analise_exploratoria.ipynb) apresenta o POP em português técnico acadêmico e
recalcula contagens por cena, altura e clima. Inclui exemplos anotados, escala dos alvos, posição
das caixas, auditoria de arquivos e comparação das partições. Não importa PyTorch ou Ultralytics.

Baixe `POP.zip` no [OSF](https://osf.io/kmcva/overview) e extraia em `projeto-1/dados`, formando
`dados/POP/train`, `dados/POP/val` e `dados/POP/test`. Há um exemplo mínimo, comentado, de download
no início do notebook. Nenhum download é executado automaticamente. O ZIP tem aproximadamente
613 MiB. Dados brutos e saídas auxiliares (`saidas-eda/`) permanecem fora do Git.

Com [uv instalado](https://docs.astral.sh/uv/getting-started/installation/), dentro de `projeto-1`,
no macOS (Intel ou Apple Silicon) ou Linux:

```bash
uv venv --python 3.12 .venv-eda
uv pip install --python .venv-eda/bin/python -r requirements-eda.txt
.venv-eda/bin/python -m jupyterlab
```

No Windows, dentro da mesma pasta, use PowerShell:

```powershell
uv venv --python 3.12 .venv-eda
uv pip install --python .venv-eda/Scripts/python.exe -r requirements-eda.txt
.\.venv-eda\Scripts\python.exe -m jupyterlab
```

Abra o notebook 00 com o kernel **Python 3** desse Jupyter. Este ambiente é independente do
`uv.lock` do treinamento. Para o notebook 01 e o treinamento, siga a seção seguinte.

O notebook distingue explicitamente os dados observados dos metadados do artigo. O pacote POP
não oferece pares RGB/térmicos alinhados nem rótulos individuais de oclusão para reproduzir todas
as figuras publicadas; essas limitações são documentadas, sem criar categorias inexistentes.

## Ambiente completo: preparação e treinamento

O ambiente já está configurado para **uv**. Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/)
e tenha aproximadamente **10 GB livres** para os dados, além do ambiente Python. O uv usa
Python 3.12 por padrão e pode baixá-lo automaticamente, se necessário.

Após clonar o repositório, os mesmos comandos funcionam no macOS, Linux e PowerShell:

```bash
git clone https://github.com/carlosfab/bootcamp-visao-computacional-drones.git
cd bootcamp-visao-computacional-drones
git switch main
cd projeto-1
uv sync --locked
uv run --locked python -m ipykernel install --sys-prefix --name pop-env --display-name "POP (torch 2.5.1)"
uv run --locked jupyter lab
```

O comando `uv sync --locked` cria `.venv` e instala as versões de `uv.lock`. Não é necessário ativar o
ambiente manualmente. O kernel é registrado dentro dele, sem alterar outros ambientes.
No Jupyter, selecione **POP (torch 2.5.1)** e abra o notebook 01.

A configuração cobre **macOS Apple Silicon**, **Linux x86_64** e **Windows x64**. No Linux e
Windows, o uv instala automaticamente o PyTorch com CUDA 12.1; no Mac, usa a distribuição
compatível com Apple Silicon. A preparação também roda em CPU. O treinamento exige GPU NVIDIA
compatível e driver apropriado; instalar o ambiente não transforma uma máquina sem GPU em uma.

Macs Intel podem executar o notebook 00 no ambiente leve acima. Para os notebooks 01 e 02,
este ambiente completo não suporta Macs Intel: não há wheel de torch 2.5.1 para macOS x86_64.
Nesse caso, execute os notebooks 01 e 02 em uma máquina Linux compatível, como o pod do guia.

`pyproject.toml` declara as dependências; `uv.lock` fixa também as versões transitivas.
Use `--locked` para impedir atualização silenciosa do lock. Os notebooks verificam o ambiente,
sem instalar pacotes nas células. Se mudar o ambiente, reinicie o kernel antes de continuar.

## Treinar com GPU

O notebook 02 exige **GPU NVIDIA com CUDA**. A referência é uma **RTX 4090 de 24 GB**; a execução
original usou aproximadamente 14,3 GB de VRAM. Considere pelo menos 16 GB, com margem para variações.
O ambiente fixado em PyTorch 2.5.1 não suporta GPUs Blackwell como a RTX 5090; use uma placa
compatível, como RTX 3090/4090, A100, A6000 ou L40S.

Se você já tem uma GPU compatível, execute `uv sync --locked` e siga no notebook 02.
Para alugar uma máquina, siga **[Configurar RunPod](GUIA-RUNPOD.md)**. Consulte o preço no
painel antes de criar recursos: some horas de GPU e armazenamento, incluindo períodos ociosos.

O Colab não faz parte do roteiro validado. Se adaptar o projeto, use armazenamento persistente e
confirme a compatibilidade do ambiente. Interrupções podem exigir retomada de `last.pt`.

## Onde ficam os dados e resultados

O notebook 00 lê `dados/POP` e grava suas tabelas e figuras em `saidas-eda`. O notebook 01
reutiliza o ZIP e a pasta POP, quando disponíveis, para evitar download e extração repetidos.
Os notebooks 01 e 02 escolhem a seguinte pasta de trabalho para a conversão e o experimento:

| Ambiente | Pasta padrão |
|---|---|
| RunPod, quando `/workspace` existe | `/workspace/projeto-pop` |
| Colab, quando `/content` existe | `/content/projeto-pop` |
| Local, abrindo o Jupyter dentro de `projeto-1` | `projeto-1/projeto-pop` |

Você pode definir a variável de ambiente `POP_RAIZ` antes de iniciar o Jupyter para escolher outra
pasta. Use o mesmo valor nos dois notebooks. Ao transportar os dados para outra máquina ou caminho,
execute novamente o notebook 01 para atualizar o YAML. O ZIP já íntegro é reutilizado.

```text
projeto-pop/
├── ambiente_preparacao.json
├── catalogo.csv
├── data/
│   ├── POP.zip
│   ├── original/
│   └── pop_yolo/                 # imagens, rótulos e pop.yaml
└── runs/<nome-da-execucao>/
    ├── ambiente_treino.json
    ├── especificacao.json
    ├── results.csv
    ├── treinamento_concluido.json # épocas efetivas, motivo de encerramento e hashes
    ├── weights/                 # best.pt, last.pt e checkpoints intermediários
    ├── modelo_congelado.pt
    ├── protocolo_congelado.json
    ├── resultado_teste.json
    └── avaliacoes/
```

No RunPod, confirme no painel que o volume de rede está anexado. A existência de `/workspace`
ou o aviso do notebook não bastam para determinar a política de persistência do armazenamento.

**Para retomar:** use a mesma configuração e o mesmo `NOME_EXECUCAO`. O notebook reaproveita o
experimento concluído, inclusive por parada antecipada, ou retoma o treinamento interrompido
a partir de `last.pt`. Preserve `treinamento_concluido.json` junto aos pesos e ao histórico.
**Para mudar parâmetros:** escolha outro `NOME_EXECUCAO` e reexecute desde o início. Isso inclui
reduzir `BATCH` por falta de memória. Preserve os resultados anteriores para comparação.

## O que entregar ao concluir

- Os três notebooks com as saídas da sua execução.
- A pasta da execução, incluindo especificação, registro do ambiente, pesos, protocolo e resultado.
- Uma análise curta: o requisito foi atingido na validação e no teste? Quais cenas e tamanhos
  concentraram os erros? Quais diferenças existem em relação ao artigo?
  Informe também o limite de épocas, a paciência e o número de épocas efetivamente executadas.

O checkpoint é selecionado pelo maior mAP50-95 na validação entre `best.pt` e `last.pt`. O limiar
operacional também vem da validação. Depois do congelamento, o teste serve para medir o resultado;
ele não deve orientar novas escolhas de parâmetros.

O POP contém pessoas em todas as imagens. Portanto, seu FPPI não determina o desempenho em
terreno vazio. Este experimento é uma avaliação educacional, não uma validação de uso operacional.

## Verificação local do código

No ambiente instalado, dentro de `projeto-1`:

```bash
uv run --locked python -m unittest discover -s tests -v
```

Essas verificações cobrem conversão de caixas, leitura de rótulos, contagem de detecções,
seleção do limiar e preservação do protocolo. Elas não substituem executar o treino na GPU.

## Referências e versões

- Song et al. (2025), [An infrared dataset for partially occluded person detection in complex
  environment for search and rescue](https://doi.org/10.1038/s41597-025-04600-0).
- [Dataset POP no OSF](https://osf.io/download/gm8u2/), baixado pelo notebook 01.
- [Integração oficial do uv com PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
- [Instalação de versões anteriores do PyTorch](https://pytorch.org/get-started/previous-versions/).
- [Tipos de armazenamento no RunPod](https://docs.runpod.io/pods/storage/types).

O projeto fixa Ultralytics 8.2.0, torch 2.5.1, torchvision 0.20.1 e NumPy 1.26.4. A versão da
Ultralytics segue a referência do artigo; o par torch/torchvision evita incompatibilidades de
carregamento de checkpoints. Não atualize essas versões durante uma execução.
