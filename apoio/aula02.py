"""Visualizações prontas para acompanhar as células da aula."""

import matplotlib.pyplot as plt
import torch
from IPython.display import HTML
from matplotlib import font_manager, rc_context
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes


def _rgb(imagem):
    """Converte PIL ou tensor CHW em tensor RGB uint8 na CPU."""
    if isinstance(imagem, Image.Image):
        return pil_to_tensor(imagem.convert("RGB"))
    if not isinstance(imagem, torch.Tensor):
        raise TypeError("Use uma imagem PIL ou um tensor CHW.")
    imagem = imagem.detach().cpu()
    if imagem.ndim != 3 or imagem.shape[0] not in (1, 3, 4):
        raise ValueError("O tensor deve ter formato [C, H, W], com C = 1, 3 ou 4.")
    if imagem.numel() == 0:
        raise ValueError("A imagem não pode estar vazia.")
    if imagem.is_floating_point():
        if not torch.isfinite(imagem).all() or imagem.min() < 0 or imagem.max() > 1:
            raise ValueError("Tensores float devem conter valores finitos entre 0 e 1.")
        imagem = (imagem * 255).round().to(torch.uint8)
    elif imagem.dtype != torch.uint8:
        raise TypeError("Use uint8 entre 0 e 255 ou float entre 0 e 1.")
    if imagem.shape[0] != 3:
        imagem = pil_to_tensor(to_pil_image(imagem).convert("RGB"))
    return imagem


def mostrar(imagem):
    """Retorna a imagem PIL para exibição na última linha da célula."""
    return to_pil_image(_rgb(imagem))


def mostrar_caixas(imagem, caixas, rotulos=None):
    """Desenha caixas xyxy fornecidas; não detecta nem filtra objetos."""
    imagem = _rgb(imagem)
    caixas = torch.as_tensor(caixas, dtype=torch.float32, device="cpu").reshape(-1, 4)
    if len(caixas) == 0:
        return to_pil_image(imagem)
    fonte = font_manager.findfont("DejaVu Sans")
    resultado = draw_bounding_boxes(
        imagem, caixas, labels=rotulos, colors="#ffbf00", width=3,
        font=fonte, font_size=18,
    )
    return to_pil_image(resultado)


def varredura(imagem, janela=256, passo=128):
    """Exibe uma janela quadrada e seu recorte, sem executar um detector."""
    imagem = _rgb(imagem).permute(1, 2, 0).numpy()
    altura, largura = imagem.shape[:2]
    if not all(isinstance(valor, int) and valor > 0 for valor in (janela, passo)):
        raise ValueError("janela e passo devem ser números inteiros positivos.")
    if janela > min(altura, largura):
        raise ValueError("A janela deve caber na altura e na largura da imagem.")
    if passo > janela:
        raise ValueError("Use passo <= janela para não deixar faixas sem cobertura.")

    def posicoes(tamanho):
        limite = tamanho - janela
        pontos = list(range(0, limite + 1, passo))
        if pontos[-1] != limite:
            pontos.append(limite)
        return pontos

    coordenadas = [(x, y) for y in posicoes(altura) for x in posicoes(largura)]
    if len(coordenadas) > 150:
        raise ValueError(
            f"São {len(coordenadas)} recortes; o limite é 150. "
            "Aumente passo/janela ou reduza a imagem. Nenhum recorte foi omitido."
        )

    with rc_context({"font.family": "DejaVu Sans"}):
        fig, eixos = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
        eixos[0].imshow(imagem)
        caixa = Rectangle((0, 0), janela, janela, fill=False, ec="#ffbf00", lw=2)
        eixos[0].add_patch(caixa)
        recorte = eixos[1].imshow(imagem[:janela, :janela])
        eixos[0].set_title("Janela de varredura", fontsize=11)
        titulo = eixos[1].set_title("", fontsize=10)
        fig.suptitle("Explorando recortes — sem predição", fontsize=11)
        for eixo in eixos:
            eixo.set_axis_off()

    def atualizar(indice):
        x, y = coordenadas[indice]
        caixa.set_xy((x - 0.5, y - 0.5))
        recorte.set_data(imagem[y:y + janela, x:x + janela])
        titulo.set_text(f"Recorte {indice + 1}/{len(coordenadas)} · x={x}, y={y}")
        return caixa, recorte, titulo

    try:
        # JPEG reduz o HTML; se necessário, reduzimos só a resolução visual.
        for dpi in (80, 60, 45):
            fig.set_dpi(dpi)
            animacao = FuncAnimation(
                fig, atualizar, frames=len(coordenadas), interval=400, repeat=False,
            )
            with rc_context({"animation.frame_format": "jpeg", "animation.embed_limit": 100}):
                html = animacao.to_jshtml(default_mode="once")
            if len(html.encode("utf-8")) < 10_000_000:
                return HTML(html)
        raise ValueError(
            "A animação excedeu 10 MB. Aumente passo/janela ou reduza a imagem. "
            "Nenhum recorte foi omitido."
        )
    finally:
        plt.close(fig)
