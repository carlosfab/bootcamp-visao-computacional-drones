# Bootcamp de Visão Computacional Aplicada a Drones

Repositório para compartilhar material com os alunos do bootcamp da Academia Sigmoidal.

![Cena sintética: câmera, campo de visão e terreno](https://raw.githubusercontent.com/carlosfab/bootcamp-visao-computacional-drones/main/dados/geometria-captura.png)

## Notebook

[Introdução à visão computacional](introducao-visao-computacional.ipynb): formação da imagem, pixels, geometria de captura e análise de uma cena aérea.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/main/introducao-visao-computacional.ipynb)

No Colab, execute as células em ordem. A primeira baixa automaticamente os dados necessários. Não é preciso enviar imagens manualmente nem ativar GPU.

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

No Windows, após clonar o repositório e entrar na pasta, use o PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Abra `introducao-visao-computacional.ipynb` no JupyterLab. A pasta `dados/` contém as imagens, máscaras e parâmetros utilizados. As imagens são sintéticas e as máscaras representam a superfície visível da pessoa.
