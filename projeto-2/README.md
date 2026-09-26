# Projeto 2 · Detecção de pessoas com RGB e térmico

Este projeto usa duas imagens da mesma cena, uma óptica e outra térmica, para detectar
pessoas com o **QFDet**. A sequência cobre exploração dos dados, preparação das entradas,
uso de pesos publicados e interpretação de previsões em vídeo. Não é necessário treinar a rede.

O último notebook compara duas versões treinadas do QFDet sobre os mesmos pares de imagens.
A atividade introduz **mudança de domínio** e a necessidade de avaliar o modelo nas condições
da aplicação. Contagens de previsões não são medidas de acurácia ou contagens de pessoas únicas.

## Sequência dos notebooks

| Notebook | Conteúdo | Ambiente |
|:--|:--|:--|
| [00 · Análise exploratória](00_analise_exploratoria.ipynb) | Pares RGB/térmico, classes, escala das pessoas e desalinhamento | EDA, CPU |
| [01 · Imagens e anotações](01_preparando_imagens_e_anotacoes.ipynb) | Coordenadas, espelhamento conjunto, normalização e canais | EDA, CPU |
| [02 · Pesos treinados](02_deteccao_com_pesos_treinados.ipynb) | QFDet no RGBTDronePerson, caixas, scores e comparação com anotações | QFDet, CPU |
| [03 · Vídeo e mudança de domínio](03_video_e_mudanca_de_dominio.ipynb) | Origem do VTUAV, pesos VTUAV-det e comparação dos dois vídeos | EDA para visualizar; QFDet para inferir |

Os notebooks incluem saídas de referência. No Notebook 03, os dois MP4 estão incorporados
ao próprio arquivo; também estão em [assets/video](assets/video). Para assisti-los, não é
necessário baixar os pesos ou os datasets. Abra o notebook em Jupyter e permita a exibição
de suas saídas HTML. Uma prévia estática pode não reproduzir vídeo; nesse caso, abra os MP4.

## Obter o material

Com Git instalado, obtenha a versão dos alunos da branch `main`:

```bash
git clone https://github.com/carlosfab/bootcamp-visao-computacional-drones.git
cd bootcamp-visao-computacional-drones/projeto-2
```

Os comandos seguintes devem ser executados nessa pasta. O projeto usa o
[uv](https://docs.astral.sh/uv/getting-started/installation/) para criar ambientes isolados.
Instale o uv antes de executar a preparação. O script pode baixar as versões necessárias do Python.

## Preparar o ambiente de exploração

```bash
python3 scripts/preparar_ambiente.py --eda
```

Essa etapa cria `.venv-eda` com Python 3.12.3, NumPy 1.26.4, pandas 2.2.3,
Matplotlib 3.9.4, Pillow 11.3.0 e JupyterLab. Não instala PyTorch.

Para obter os dados dos Notebooks 00 e 01:

```bash
.venv-eda/bin/python suporte/preparar_dados.py --rgbtdroneperson
```

O comando baixa o ZIP **RGBTDronePerson** e os dois JSONs térmicos completos da fonte oficial,
verifica sua integridade e extrai os arquivos. O ZIP tem aproximadamente **1,57 GB**;
reserve pelo menos **4 GB** para esse conjunto, além do espaço dos ambientes Python.
Arquivos já verificados são reutilizados. O ZIP não contém os JSONs COCO, que são obtidos separadamente.

Abra o Jupyter:

```bash
.venv-eda/bin/python -m jupyterlab
```

Selecione **Projeto 2 - EDA** e execute os Notebooks 00 e 01, cada um do início ao fim.

## Preparar a inferência QFDet

A inferência usa uma pilha mais antiga, separada do ambiente de exploração e dos demais
projetos do bootcamp. **Não use o `requirements.txt` da raiz nem atualize as versões do detector.**

```bash
python3 scripts/preparar_ambiente.py --qfdet
```

Execute depois da preparação EDA. A etapa cria `.venv-qfdet-cpu`, obtém a implementação
oficial em `codigo/qfdet` no commit fixado e registra o kernel no Jupyter do ambiente EDA.
As dependências centrais são Python 3.10.11, torch 1.13.1, torchvision 0.14.1,
NumPy 1.23.5 e **mmcv-full 1.6.1**. Linux usa as distribuições CPU correspondentes do PyTorch.

O MMCV contém operações compiladas em C++. No macOS Intel, são necessárias as ferramentas
de desenvolvimento do Xcode (`xcode-select --install`, caso ainda não estejam instaladas).
No Linux x86_64, é necessário um compilador C++ e as ferramentas de construção da distribuição.
A compilação pode levar alguns minutos. O script preserva a versão do MMCV e verifica uma
operação NMS em CPU ao final; ele não substitui silenciosamente essa biblioteca por uma versão recente.

O roteiro foi executado em **macOS Intel**. O caminho Linux x86_64 está incluído, mas não foi
executado nesta preparação. Windows, Apple Silicon e Colab não fazem parte do ambiente de
inferência validado. Os vídeos e as saídas salvas podem ser consultados sem instalar o detector.

Abra novamente o Jupyter pelo ambiente EDA e selecione **Projeto 2 - QFDet CPU** no
Notebook 02 e na reexecução opcional do Notebook 03. Para executar a comparação pelos registros
no Notebook 03, o kernel EDA é suficiente. Os notebooks são independentes do estado em memória um do outro.

## Pesos publicados pelos autores

O Notebook 02 usa o checkpoint treinado em **RGBTDronePerson**. O Notebook 03 usa o checkpoint
treinado em **VTUAV-det**, com a configuração correspondente. Ambos têm aproximadamente 485 MB.
As células reutilizam o arquivo verificado ou fazem seu download quando ele está ausente.
No Notebook 03, isso só acontece ao ativar a reexecução e ter os 89 pares disponíveis.

Também é possível obtê-los antes de abrir os notebooks:

```bash
.venv-qfdet-cpu/bin/python suporte/preparar_dados.py --pesos rgbtdroneperson
.venv-qfdet-cpu/bin/python suporte/preparar_dados.py --pesos vtuav
```

- [Checkpoint RGBTDronePerson](https://drive.google.com/file/d/1TuVXy_h0PxTK8qLY1MgqJTNiAumV8p0V/view).
- [Checkpoint VTUAV-det](https://drive.google.com/file/d/1Savf3oeiWek4eeXrvYLuaoBMoZW3nag8/view).
- [Código e configurações QFDet](https://github.com/NNNNerd/mmdet-rgbtdroneperson).

Os hashes em [suporte/fontes.json](suporte/fontes.json) foram calculados nos arquivos usados
para preparar esta atividade. Eles identificam os arquivos esperados; não são assinaturas publicadas
pelos autores. Cotas ou mudanças nos serviços de hospedagem podem interromper downloads.
Um erro de integridade ou um arquivo parcial interrompe a preparação, sem substituição silenciosa.
Os downloads escrevem diretamente em `<destino>.part`, inclusive se houver Ctrl+C.
Temporários de execuções anteriores, como `train_ST_004.zip.part<aleatório>.part`, também bloqueiam
uma nova transferência. O erro informa os caminhos exatos. Confira a causa da interrupção e remova
manualmente apenas os parciais indicados antes de tentar de novo; não há retomada automática.

## Reexecutar o trecho VTUAV

**O Notebook 03 inicia em modo de visualização** (`REEXECUTAR_INFERENCIA = False`).
“Run All” apresenta os vídeos, a figura de referência e as contagens salvas sem baixar pesos ou dados
nem carregar o detector. Esse modo usa o ambiente EDA e é a alternativa para quem não tem o ambiente
QFDet validado. Também é possível apenas ler as saídas salvas e abrir os MP4 sem executar Python.

Para recalcular as previsões, prepare o ambiente QFDet, selecione seu kernel, obtenha os dados com
o comando abaixo e mude a opção para `REEXECUTAR_INFERENCIA = True` antes de executar do início.
Se faltarem imagens do recorte, o notebook informa como prepará-las e mantém a comparação pelos registros.
O recorte não está incluído como imagens brutas no Git.

```bash
.venv-qfdet-cpu/bin/python suporte/preparar_dados.py --vtuav
```

O comando obtém o pacote oficial **`train_ST_004.zip`, de 18,85 GB**, e extrai somente
os 89 pares utilizados na demonstração. Reserve pelo menos **20 GB livres adicionais**
para esse pacote e o recorte. O download depende da disponibilidade e da cota do provedor.
Não é necessário baixar os demais pacotes do VTUAV.

Se baixar o ZIP manualmente pela [fonte oficial](https://drive.google.com/file/d/1UdbUJlJaTB7loptBlBY2Jb47YVDIxrbj/view),
coloque-o em `dados/archives/train_ST_004.zip` e execute o mesmo comando. A extração confere o CRC
e os hashes individuais dos pares. Um recorte já presente e íntegro é reutilizado sem obter o ZIP novamente.

O Notebook 03 repete a inferência VTUAV-det nos 89 pares e confere as previsões contra os
registros da demonstração. A execução em CPU leva alguns minutos; a duração de 12 segundos
do MP4 não mede processamento em tempo real. A versão RGBTDronePerson é comparada por seus
registros salvos, produzidos com as mesmas imagens.

## Organização dos arquivos

```text
projeto-2/
├── 00_analise_exploratoria.ipynb
├── 01_preparando_imagens_e_anotacoes.ipynb
├── 02_deteccao_com_pesos_treinados.ipynb
├── 03_video_e_mudanca_de_dominio.ipynb
├── assets/                     # diagramas, vídeos e previsões da comparação
├── scripts/preparar_ambiente.py
├── suporte/                    # obtenção dos arquivos e carregamento verificado
├── codigo/qfdet/                # código oficial obtido pelo setup; fora do Git
├── dados/                      # datasets e recorte; fora do Git
├── modelos/                    # checkpoints; fora do Git
└── resultados/                 # resultados próprios; fora do Git
```

Preserve as divisões originais entre treino e validação. As caixas previstas e as anotações
completas usam coordenadas térmicas; copiá-las para o RGB não corrige o registro entre sensores.
O checkpoint VTUAV-det possui três canais físicos nos classificadores, mas somente a saída zero
tem o rótulo `person`. As demais saídas não devem ser interpretadas como `rider` ou `crowd`.

Os vídeos são demonstrações qualitativas. Usam amostragem temporal variável e detecções independentes,
sem rastreamento de identidades. A comparação preserva a configuração correspondente a cada checkpoint,
incluindo normalização e NMS; ela não isola o efeito dos pesos nem reproduz um benchmark.
`reweight` é um parâmetro da função de perda no treinamento, preservado na configuração por fidelidade;
a flag não é consultada na inferência. Consulte as [notas técnicas](NOTAS-TECNICAS.md) sobre artigo e implementação.

## Referências e atribuição

- Zhang, Y. et al. (2023), [Drone-based RGBT tiny person detection](https://doi.org/10.1016/j.isprsjprs.2023.08.016).
- [RGBTDronePerson e descrição do VTUAV-det](https://nnnnerd.github.io/RGBTDronePerson/).
- Zhang, P. et al. (2022), [Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline](https://zhang-pengyu.github.io/DUT-VTUAV/).
- [Implementação oficial QFDet](https://github.com/NNNNerd/mmdet-rgbtdroneperson), commit `a79210c283597294ded255028bb8fe16e1c241cd`.

Consulte [ATRIBUICOES.md](ATRIBUICOES.md) para a origem dos dados, dos vídeos e das implementações.
