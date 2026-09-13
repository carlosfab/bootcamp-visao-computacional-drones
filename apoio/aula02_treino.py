"""Dados e visualizações para a introdução ao treinamento de uma CNN."""

from copy import deepcopy
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.display import HTML
from matplotlib import rc_context
from PIL import Image
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import to_pil_image


ESTILO = {"font.family": "DejaVu Sans", "font.size": 11}


def rede_exemplo():
    """Retorna uma arquitetura pronta, com pesos aleatórios e sem treinamento."""
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 14 * 14, 10),
    )


def resumo(rede):
    """Exibe formas e parâmetros de uma rede Sequential sem alterar a original."""
    if not isinstance(rede, nn.Sequential):
        raise TypeError("Use uma rede construída com nn.Sequential.")
    copia = deepcopy(rede).cpu().eval()
    parametro = next(copia.parameters())
    saida = torch.zeros(1, 1, 28, 28, dtype=parametro.dtype)
    linhas = [["Entrada", "Imagem em escala de cinza", str(tuple(saida.shape)), "—"]]
    with torch.inference_mode():
        for nome, camada in copia.named_children():
            saida = camada(saida)
            quantidade = sum(p.numel() for p in camada.parameters())
            linhas.append([
                f"{nome} · {type(camada).__name__}", camada.extra_repr(),
                str(tuple(saida.shape)), f"{quantidade:,}".replace(",", "."),
            ])
    total = sum(p.numel() for p in copia.parameters())
    linhas.append(["Total", "", "", f"{total:,}".replace(",", ".")])
    estilo = 'style="padding:9px 12px;border-bottom:1px solid #dce1e6;text-align:left"'
    colunas = ["Camada", "Configuração", "Saída (N = 1)", "Parâmetros"]
    cabecalho = "".join(f"<th {estilo}>{coluna}</th>" for coluna in colunas)
    corpo = "".join(
        "<tr>" + "".join(f"<td {estilo}>{escape(valor)}</td>" for valor in linha) + "</tr>"
        for linha in linhas
    )
    return HTML(
        '<table style="font-family:DejaVu Sans,Arial,sans-serif;border-collapse:collapse">'
        f"<thead><tr>{cabecalho}</tr></thead><tbody>{corpo}</tbody></table>"
    )


def preparar_dados(root="dados"):
    """Baixa o Fashion-MNIST oficial via HTTPS, validando os arquivos com MD5."""
    pasta = Path(root) / "FashionMNIST" / "raw"
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    for nome, md5 in FashionMNIST.resources:
        if not (pasta / nome.removesuffix(".gz")).exists():
            download_and_extract_archive(base + nome, str(pasta), filename=nome, md5=md5)


def mostrar(imagem):
    """Retorna uma imagem PIL ampliada para exibição na última linha da célula."""
    if isinstance(imagem, torch.Tensor):
        imagem = to_pil_image(imagem.detach().cpu())
    if not isinstance(imagem, Image.Image):
        raise TypeError("Use uma imagem PIL ou um tensor de imagem.")
    return imagem.resize((224, 224), Image.Resampling.NEAREST)


def amostras(dataset, classes):
    """Mostra os dez primeiros exemplos do conjunto, na ordem recebida."""
    with rc_context(ESTILO):
        fig, eixos = plt.subplots(2, 5, figsize=(10, 4.6), layout="constrained")
        for indice, eixo in enumerate(eixos.flat):
            imagem, rotulo = dataset[indice]
            eixo.imshow(mostrar(imagem), cmap="gray", vmin=0, vmax=255)
            eixo.set_title(str(classes[int(rotulo)]), fontsize=11)
            eixo.set_axis_off()
    plt.close(fig)
    return fig


def curva(perdas):
    """Mostra a perda média de treinamento ao final de cada época."""
    epocas = range(1, len(perdas) + 1)
    with rc_context(ESTILO):
        fig, eixo = plt.subplots(figsize=(7, 3.6), layout="constrained")
        eixo.plot(epocas, perdas, marker="o", color="#1976b8", linewidth=2)
        eixo.set_xlabel("Época")
        eixo.set_ylabel("Perda média de treino")
        eixo.set_xticks(list(epocas))
        eixo.grid(axis="y", alpha=0.2)
        eixo.spines[["top", "right"]].set_visible(False)
    plt.close(fig)
    return fig
