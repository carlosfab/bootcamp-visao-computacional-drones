"""Visualizações de matrizes, mapas e convolução para a aula."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from IPython.display import HTML
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def _tensor(valores):
    tensor = torch.as_tensor(valores).detach().cpu().float()
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("Use valores finitos em um tensor não vazio.")
    return tensor


def _plano(valores):
    tensor = _tensor(valores)
    if tensor.ndim != 2:
        raise ValueError("Use uma matriz 2D: [altura, largura].")
    return tensor


def _numero(valor):
    valor = round(float(valor), 2)
    return f"{valor:.2f}".rstrip("0").rstrip(".") if valor else "0"


def _grade(eixo, altura, largura):
    eixo.set_xticks(range(largura))
    eixo.set_yticks(range(altura))
    eixo.tick_params(length=0, labelsize=8, colors="#626975")
    eixo.set_xticks(torch.arange(largura + 1).numpy() - 0.5, minor=True)
    eixo.set_yticks(torch.arange(altura + 1).numpy() - 0.5, minor=True)
    eixo.grid(which="minor", color="#bfc5ce", linewidth=0.6)
    eixo.tick_params(which="minor", length=0)
    for borda in eixo.spines.values():
        borda.set_visible(False)


def mostrar(imagem):
    """Retorna PIL; tensores podem ser uint8 ou float entre 0 e 1."""
    if isinstance(imagem, Image.Image):
        return imagem.convert("RGB")
    imagem = torch.as_tensor(imagem).detach().cpu()
    if imagem.ndim not in (2, 3) or (imagem.ndim == 3 and imagem.shape[0] not in (1, 3, 4)):
        raise ValueError("Use um tensor [H, W] ou [C, H, W], com C = 1, 3 ou 4.")
    if imagem.numel() == 0:
        raise ValueError("A imagem não pode estar vazia.")
    if imagem.is_floating_point():
        if not torch.isfinite(imagem).all() or imagem.min() < 0 or imagem.max() > 1:
            raise ValueError("Para mostrar uma imagem float, use valores entre 0 e 1.")
        imagem = (imagem * 255).round().to(torch.uint8)
    elif imagem.dtype != torch.uint8:
        raise TypeError("Use uint8 ou float entre 0 e 1; para outros valores, use matriz/mapas.")
    return to_pil_image(imagem).convert("RGB")


def matriz(valores):
    """Retorna uma figura com os números de uma matriz de até 10 × 10."""
    valores = _plano(valores)
    altura, largura = valores.shape
    if max(altura, largura) > 10:
        raise ValueError("Para ler os números, use uma matriz de até 10 × 10.")
    limite = max(float(valores.abs().max()), 1e-6)
    with rc_context({"font.family": "DejaVu Sans"}):
        fig, eixo = plt.subplots(figsize=(max(3, largura * 0.55), max(3, altura * 0.55)))
        eixo.imshow(valores, cmap="RdBu_r", vmin=-limite, vmax=limite)
        _grade(eixo, altura, largura)
        for y in range(altura):
            for x in range(largura):
                cor = "white" if abs(float(valores[y, x])) > limite * 0.6 else "#202631"
                eixo.text(x, y, _numero(valores[y, x]), ha="center", va="center", color=cor)
        fig.tight_layout()
    plt.close(fig)
    return fig


def _painel(valores, titulos):
    limite = max(max(float(valor.abs().max()) for valor in valores), 1e-6)
    colunas = min(len(valores), 4)
    linhas = (len(valores) + colunas - 1) // colunas
    with rc_context({"font.family": "DejaVu Sans"}):
        fig, eixos = plt.subplots(linhas, colunas, figsize=(3 * colunas, 2.8 * linhas),
                                 squeeze=False, layout="constrained")
        for eixo, valor, titulo in zip(eixos.flat, valores, titulos):
            desenho = eixo.imshow(valor, cmap="RdBu_r", vmin=-limite, vmax=limite)
            eixo.set_title(titulo, fontsize=11)
            eixo.set_axis_off()
        for eixo in list(eixos.flat)[len(valores):]:
            eixo.set_axis_off()
        fig.colorbar(desenho, ax=list(eixos.flat), shrink=0.8, label="Valor")
    plt.close(fig)
    return fig


def comparar(antes, depois):
    """Retorna dois mapas com a mesma escala, simétrica em torno de zero."""
    return _painel([_plano(antes), _plano(depois)], ["Antes", "Depois"])


def mapas(tensor):
    """Retorna até oito canais; aceita [1, C, H, W], [C, H, W] ou [H, W]."""
    tensor = _tensor(tensor)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor[None]
    if tensor.ndim != 3:
        raise ValueError("Use [1, C, H, W], [C, H, W] ou [H, W].")
    if len(tensor) > 8:
        raise ValueError("Mostre até 8 canais por vez; selecione os canais desejados antes.")
    return _painel(list(tensor), [f"Canal {canal}" for canal in range(len(tensor))])


def convolucao(recorte, filtro):
    """Anima stride 1, padding 0 e bias 0; o filtro não é invertido, como em conv2d."""
    entrada, filtro = _plano(recorte), _plano(filtro)
    if tuple(filtro.shape) != (3, 3) or min(entrada.shape) < 3 or max(entrada.shape) > 10:
        raise ValueError("Use um filtro 3 × 3 e uma entrada com cada dimensão entre 3 e 10.")
    resultado = F.conv2d(entrada[None, None], filtro[None, None], bias=None)[0, 0]
    altura, largura = resultado.shape
    coordenadas = [(y, x) for y in range(altura) for x in range(largura)]
    if len(coordenadas) > 64:
        raise ValueError("A animação aceita até 64 posições; reduza o recorte.")
    limite = max(float(resultado.abs().max()), 1e-6)
    with rc_context({"font.family": "DejaVu Sans"}):
        fig, eixos = plt.subplots(1, 3, figsize=(11.5, 4.2))
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.23, top=0.8, wspace=0.3)
        fig.suptitle("Convolução no PyTorch · stride = 1 · padding = 0 · bias = 0", fontsize=12)
        eixos[0].imshow(entrada, cmap="gray", vmin=float(entrada.min()), vmax=float(entrada.max()) + 1e-6)
        eixos[0].set_title("Entrada e janela 3 × 3", fontsize=11)
        _grade(eixos[0], *entrada.shape)
        meio = (float(entrada.min()) + float(entrada.max())) / 2
        for y in range(entrada.shape[0]):
            for x in range(entrada.shape[1]):
                cor = "white" if float(entrada[y, x]) < meio else "#202631"
                eixos[0].text(x, y, _numero(entrada[y, x]), ha="center", va="center", color=cor, fontsize=9)
        janela = Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="#ffbf00", lw=3)
        eixos[0].add_patch(janela)
        eixos[1].imshow(torch.zeros(3, 3), cmap="Greys", vmin=0, vmax=1)
        eixos[1].set_title("Valor × peso do filtro", fontsize=11)
        _grade(eixos[1], 3, 3)
        produtos = [eixos[1].text(x, y, "", ha="center", va="center", fontsize=10)
                    for y in range(3) for x in range(3)]
        soma = eixos[1].text(0.5, -0.19, "", transform=eixos[1].transAxes, ha="center", fontsize=11)
        desenho = eixos[2].imshow(torch.full_like(resultado, float("nan")), cmap="RdBu_r", vmin=-limite, vmax=limite)
        eixos[2].set_title("Saída: uma soma por posição", fontsize=11)
        _grade(eixos[2], altura, largura)
        numeros = [eixos[2].text(x, y, "", ha="center", va="center", fontsize=10) for y, x in coordenadas]
        posicao = Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="#ffbf00", lw=3)
        eixos[2].add_patch(posicao)
        contador = fig.text(0.5, 0.035, "", ha="center", fontsize=10, color="#626975")

    def atualizar(indice):
        y, x = coordenadas[indice]
        janela.set_xy((x - 0.5, y - 0.5))
        trecho = entrada[y:y + 3, x:x + 3]
        produto = trecho * filtro
        for texto, valor, peso, multiplicacao in zip(produtos, trecho.flatten(), filtro.flatten(), produto.flatten()):
            texto.set_text(f"{_numero(valor)} × {_numero(peso)}\n= {_numero(multiplicacao)}")
        soma.set_text(f"Soma dos 9 produtos = {_numero(produto.sum())}")
        parcial = torch.full_like(resultado, float("nan"))
        parcial.flatten()[:indice + 1] = resultado.flatten()[:indice + 1]
        desenho.set_data(parcial)
        posicao.set_xy((x - 0.5, y - 0.5))
        for n, texto in enumerate(numeros):
            valor = resultado.flatten()[n]
            texto.set_text(_numero(valor) if n <= indice else "")
            texto.set_color("white" if abs(float(valor)) > limite * 0.6 else "#202631")
        contador.set_text(f"Posição {indice + 1}/{len(coordenadas)} · saída[{y}, {x}]")

    try:
        for dpi in (90, 70, 55):
            fig.set_dpi(dpi)
            animacao = FuncAnimation(fig, atualizar, frames=len(coordenadas), interval=700, repeat=False)
            with rc_context({"animation.frame_format": "png", "animation.embed_limit": 50}):
                html = animacao.to_jshtml(default_mode="once")
            if len(html.encode("utf-8")) < 5_000_000:
                return HTML(html)
        raise ValueError("A animação excedeu 5 MB. Reduza o recorte; nenhuma posição foi omitida.")
    finally:
        plt.close(fig)
