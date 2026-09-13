"""Recortes e figuras para examinar saídas reais de um detector."""

import math
import textwrap

import matplotlib.pyplot as plt
import torch
from matplotlib import rc_context
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def _imagem(imagem):
    if isinstance(imagem, Image.Image):
        return pil_to_tensor(imagem.convert("RGB"))
    imagem = torch.as_tensor(imagem).detach().cpu()
    if imagem.ndim != 3 or imagem.shape[0] != 3 or imagem.numel() == 0:
        raise ValueError("Use uma imagem RGB no formato [3, altura, largura].")
    if imagem.is_floating_point():
        if not torch.isfinite(imagem).all() or imagem.min() < 0 or imagem.max() > 1:
            raise ValueError("A imagem float deve conter valores finitos entre 0 e 1.")
        imagem = (imagem * 255).round().to(torch.uint8)
    elif imagem.dtype != torch.uint8:
        raise TypeError("Use uma imagem uint8 ou float entre 0 e 1.")
    return imagem


def _celulas(imagem, tamanho):
    if not isinstance(tamanho, int) or tamanho <= 0:
        raise ValueError("O tamanho do recorte deve ser um inteiro positivo.")
    altura, largura = imagem.shape[-2:]
    return [(x, y, min(tamanho, largura - x), min(tamanho, altura - y))
            for y in range(0, altura, tamanho) for x in range(0, largura, tamanho)]


def _limiar(limiar):
    if not isinstance(limiar, (int, float)) or not math.isfinite(limiar) or not 0 <= limiar <= 1:
        raise ValueError("O limiar deve estar entre 0 e 1.")
    return f"Classes detectadas em cada recorte · score ≥ {limiar:g}".replace(".", ",")


def _rotulos(rotulos, quantidade):
    if len(rotulos) != quantidade or any(isinstance(item, str) for item in rotulos):
        raise ValueError("Forneça uma lista de classes para cada recorte, na ordem da grade.")
    if any(not isinstance(nome, str) for item in rotulos for nome in item):
        raise TypeError("Cada classe deve ser representada pelo seu nome em texto.")
    return [list(item) for item in rotulos]


def _texto(nomes, largura=23):
    return "\n".join(textwrap.fill(nome, width=largura, break_long_words=False,
                                  break_on_hyphens=False) for nome in nomes) if nomes else "sem detecção"


def _grade(eixo, imagem, tamanho):
    eixo.imshow(imagem.permute(1, 2, 0).numpy())
    for numero, (x, y, largura, altura) in enumerate(_celulas(imagem, tamanho), start=1):
        for cor, espessura in [("#101820", 3.5), ("#ffce33", 1.4)]:
            eixo.add_patch(Rectangle((x - 0.5, y - 0.5), largura, altura,
                                     fill=False, ec=cor, lw=espessura))
        eixo.text(x + 8, y + 8, str(numero), va="top", color="white", fontsize=10,
                  weight="bold", bbox={"facecolor": "#101820", "edgecolor": "none", "pad": 2})
    eixo.set_axis_off()


def _seta(fig, inicio, fim):
    fig.add_artist(FancyArrowPatch(inicio, fim, transform=fig.transFigure,
                                  arrowstyle="->", mutation_scale=15, lw=1.4, color="#4b5563"))


def recortar_grade(imagem, tamanho=256):
    """Retorna recortes CHW uint8 por linha; bordas incompletas geram recortes menores."""
    imagem = _imagem(imagem)
    return [imagem[:, y:y + altura, x:x + largura].contiguous()
            for x, y, largura, altura in _celulas(imagem, tamanho)]


def mostrar_grade(imagem, tamanho=256):
    """Retorna uma figura com a divisão da imagem e os números dos recortes."""
    imagem = _imagem(imagem)
    with rc_context({"font.family": "DejaVu Sans"}):
        fig, eixo = plt.subplots(figsize=(9, 6), layout="constrained")
        _grade(eixo, imagem, tamanho)
    plt.close(fig)
    return fig


def resumir_recortes(saidas, categorias, limiar=0.5):
    """Extrai classes únicas com score >= limiar; uma lista vazia indica sem detecção."""
    _limiar(limiar)
    resumo = []
    for saida in saidas:
        scores = torch.as_tensor(saida["scores"]).detach().cpu()
        labels = torch.as_tensor(saida["labels"]).detach().cpu()
        if scores.ndim != 1 or labels.shape != scores.shape or not torch.isfinite(scores).all():
            raise ValueError("Cada saída deve ter scores e labels 1D com o mesmo tamanho.")
        if labels.is_floating_point() and not torch.equal(labels, labels.round()):
            raise ValueError("Os identificadores de classes devem ser inteiros.")
        nomes = []
        for indice in labels[scores >= limiar].tolist():
            indice = int(indice)
            if indice < 0 or indice >= len(categorias):
                raise ValueError(f"A classe {indice} não está na lista de categorias.")
            nome = categorias[indice]
            if nome not in nomes:
                nomes.append(nome)
        resumo.append(nomes)
    return resumo


def mostrar_mapa(imagem, rotulos, tamanho=256, limiar=0.5):
    """Retorna a cena e a grade de classes; as cores indicam somente presença de detecção."""
    imagem = _imagem(imagem)
    celulas = _celulas(imagem, tamanho)
    rotulos = _rotulos(rotulos, len(celulas))
    legenda = _limiar(limiar)
    textos = [_texto(nomes, largura=18) for nomes in rotulos]
    altura, largura = imagem.shape[-2:]
    escala = max(((texto.count("\n") + 1) * 0.17 + 0.32) / celula[3]
                 for celula, texto in zip(celulas, textos))
    largura_fig = max(13, largura * escala / 0.425)
    altura_fig = max(5, altura * escala / 0.62)
    with rc_context({"font.family": "DejaVu Sans"}):
        fig = plt.figure(figsize=(largura_fig, altura_fig))
        fig.suptitle("Inferência independente por recorte", fontsize=14, y=0.98)
        cena = fig.add_axes([0.025, 0.22, 0.425, 0.62])
        _grade(cena, imagem, tamanho)
        cena.set_title("Cena dividida em recortes", fontsize=11, pad=10)
        mapa = fig.add_axes([0.55, 0.22, 0.425, 0.62])
        mapa.set(xlim=(-0.5, largura - 0.5), ylim=(altura - 0.5, -0.5), aspect="equal")
        mapa.set_title("Classes detectadas em cada recorte", fontsize=11, pad=10)
        mapa.set_axis_off()
        for numero, (celula, nomes, texto) in enumerate(zip(celulas, rotulos, textos), start=1):
            x, y, w, h = celula
            cor = "#dfebf7" if nomes else "#f2f3f5"
            mapa.add_patch(Rectangle((x - 0.5, y - 0.5), w, h, fc=cor, ec="#677487", lw=1))
            mapa.text(x + 10, y + 10, str(numero), va="top", color="#526176", fontsize=9)
            mapa.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=10)
        _seta(fig, (0.47, 0.53), (0.53, 0.53))
        fig.legend(handles=[Patch(fc="#dfebf7", label="Com detecção"),
                            Patch(fc="#f2f3f5", ec="#b5bdc8", label="Sem detecção")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=2, frameon=False, fontsize=9)
        fig.text(0.5, 0.06, legenda, ha="center", fontsize=10, color="#4b5563")
    plt.close(fig)
    return fig
