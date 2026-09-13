# Bootcamp de Visão Computacional Aplicada a Drones

Material de apoio para acompanhar as práticas do bootcamp da Academia Sigmoidal.

## Aula 2 · PyTorch e detecção de objetos

Os notebooks estão na pasta [aula-02](aula-02). Siga a sequência abaixo. Cada um pode ser executado de forma independente.

| Ordem | Notebook | Executar |
|---|---|---|
| 01 | [Detecção de objetos](aula-02/01-deteccao-objetos.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/01-deteccao-objetos.ipynb) |
| 02 | [Mapas de características](aula-02/02-mapas-caracteristicas.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/02-mapas-caracteristicas.ipynb) |
| 03 | [Saídas de um detector](aula-02/03-saidas-detector.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/aula-02/03-saidas-detector.ipynb) |

### Como executar no Colab

1. Clique em **Abrir no Colab** no notebook desejado.
2. Execute a preparação e siga as células em ordem. Para executar todas, use **Ambiente de execução → Executar tudo**.
3. Para guardar suas alterações, salve uma cópia no Google Drive.

As imagens e funções auxiliares são baixadas automaticamente por URLs HTTPS absolutas. Não é necessário enviar arquivos manualmente. Os notebooks de detecção baixam os pesos do modelo no primeiro carregamento. GPU é opcional; as inferências demoram mais em CPU.

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
