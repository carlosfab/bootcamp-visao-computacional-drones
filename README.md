# Bootcamp de Visão Computacional Aplicada a Drones

Material de apoio para acompanhar as práticas do bootcamp da Academia Sigmoidal.

## Projeto 3 · Detecção, rastreamento e contagem de veículos

A pasta **[projeto-3](projeto-3/README.md)** contém os fundamentos ilustrados e três
notebooks executados: Supervision e coordenadas, tracking com ByteTrack e projeto
final de contagem na rotatória. O README do projeto contém links diretos para abrir
cada notebook no Colab. Os dados são baixados pelas células de preparação. O projeto
utiliza um ambiente próprio, com dependências e instruções documentadas na pasta.

## Projeto 1 · Busca e salvamento com imagens térmicas

A pasta **[projeto-1](projeto-1/README.md)** contém o experimento completo: preparação do
ambiente e dos dados, treinamento do detector, avaliação e análise dos resultados.
Comece pelo README da pasta e execute os notebooks 00, 01 e 02 na ordem indicada.
A análise exploratória (00) usa um ambiente leve, sem PyTorch ou GPU. O projeto usa
um ambiente próprio; o ambiente uv está definido em `projeto-1/pyproject.toml` e `projeto-1/uv.lock`.

## Aula 2 · PyTorch e visão computacional

Os notebooks estão na pasta [aula-02](aula-02). Siga a sequência abaixo. Cada um pode ser executado de forma independente.

| Ordem | Notebook | Executar |
|---|---|---|
| 01 | [Detecção de objetos](aula-02/01-deteccao-objetos.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/01-deteccao-objetos.ipynb) |
| 02 | [Mapas de características](aula-02/02-mapas-caracteristicas.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/02-mapas-caracteristicas.ipynb) |
| 03 | [Saídas de um detector](aula-02/03-saidas-detector.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/03-saidas-detector.ipynb) |
| 04 | [Treinamento e inferência](aula-02/04-treinamento-inferencia.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/04-treinamento-inferencia.ipynb) |

### Como executar no Colab

1. Clique em **Abrir no Colab** no notebook desejado.
2. Execute a preparação e siga as células em ordem. Para executar todas, use **Ambiente de execução → Executar tudo**.
3. Para guardar suas alterações, salve uma cópia no Google Drive.

As imagens e funções auxiliares são baixadas automaticamente por URLs HTTPS absolutas. Não é necessário enviar arquivos manualmente. Os notebooks de detecção baixam os pesos do modelo no primeiro carregamento. GPU é opcional; as inferências demoram mais em CPU. O notebook 04 treina uma CNN pequena em CPU com imagens do Fashion-MNIST, baixadas do repositório oficial.

## Ambiente local

Com Python 3.12 e Git instalados, no macOS ou Linux:

```bash
git clone https://github.com/carlosfab/bootcamp-visao-computacional-drones.git
cd bootcamp-visao-computacional-drones
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

No Windows, após clonar e entrar na pasta, use o PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Abra um notebook da pasta `aula-02`. Os downloads das células de preparação também funcionam no ambiente local.

## Material complementar

[Introdução à visão computacional](introducao-visao-computacional.ipynb): formação da imagem, pixels e geometria de captura.
