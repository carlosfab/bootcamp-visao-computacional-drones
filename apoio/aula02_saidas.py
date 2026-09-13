"""Visualizações dos dados fornecidos pelo aluno; não executa inferência."""

from html import escape

import matplotlib.pyplot as plt
import torch
from IPython.display import HTML
from matplotlib import font_manager, rc_context
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes


COR_CAIXA = "#ffbf00"
ESTILO = {"font.family": "DejaVu Sans", "font.size": 11}


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


def _caixas(caixas):
    """Obtém caixas xyxy na CPU sem selecionar nem reordenar linhas."""
    caixas = torch.as_tensor(caixas).detach().to(device="cpu", dtype=torch.float32)
    if caixas.numel() == 0:
        return caixas.reshape(0, 4)
    caixas = caixas.reshape(-1, 4)
    if not torch.isfinite(caixas).all() or (caixas[:, 2:] < caixas[:, :2]).any():
        raise ValueError("Forneça caixas xyxy finitas, com x2 >= x1 e y2 >= y1.")
    return caixas


def mostrar(imagem):
    """Retorna a imagem PIL para exibição na última linha da célula."""
    return to_pil_image(_rgb(imagem))


def mostrar_caixas(imagem, caixas, rotulos=None):
    """Desenha exatamente as caixas xyxy recebidas, sem detectar nem filtrar."""
    imagem, caixas = _rgb(imagem), _caixas(caixas)
    if rotulos is not None and len(rotulos) != len(caixas):
        raise ValueError("Forneça um rótulo para cada caixa.")
    if len(caixas) == 0:
        return to_pil_image(imagem)
    resultado = to_pil_image(draw_bounding_boxes(imagem, caixas, colors=COR_CAIXA, width=3))
    if rotulos is None:
        return resultado

    desenho = ImageDraw.Draw(resultado)
    largura, altura = resultado.size
    caminho_fonte = font_manager.findfont("DejaVu Sans")
    margem = 4
    for caixa, rotulo in zip(caixas.tolist(), rotulos):
        if not rotulo:
            continue
        fonte = ImageFont.truetype(caminho_fonte, 18)
        limites = desenho.textbbox((0, 0), rotulo, font=fonte)
        while (limites[2] - limites[0] + 2 * margem > largura
               or limites[3] - limites[1] + 2 * margem > altura) and fonte.size > 1:
            fonte = ImageFont.truetype(caminho_fonte, fonte.size - 1)
            limites = desenho.textbbox((0, 0), rotulo, font=fonte)
        texto_largura = limites[2] - limites[0] + 2 * margem
        texto_altura = limites[3] - limites[1] + 2 * margem
        x = max(0, min(round(caixa[0]), largura - texto_largura))
        y = round(caixa[1]) - texto_altura
        if y < 0:
            y = round(caixa[1]) + 3
        y = max(0, min(y, altura - texto_altura))
        desenho.rectangle((x, y, x + texto_largura - 1, y + texto_altura - 1), fill="#202830")
        desenho.text((x + margem - limites[0], y + margem - limites[1]),
                     rotulo, font=fonte, fill="white")
    return resultado


def comparar_recortes(imagem):
    """Mostra três janelas manuais na praça de 768 × 512 pixels e seus recortes."""
    imagem = _rgb(imagem).permute(1, 2, 0).numpy()
    if imagem.shape[:2] != (512, 768):
        raise ValueError("Este exemplo usa a imagem da praça com 768 × 512 pixels.")
    janelas = [(350, 150, 64), (280, 115, 192), (232, 60, 288)]
    titulos = ["Objeto parcial", "Objeto completo e contexto", "Contexto ampliado"]
    cores = ["#ffbf00", "#00c6e0", "#e069ff"]

    with rc_context(ESTILO):
        fig, eixos = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
        for coluna, ((x, y, lado), titulo, cor) in enumerate(zip(janelas, titulos, cores)):
            origem, destino = eixos[:, coluna]
            origem.imshow(imagem, extent=(0, 768, 512, 0))
            origem.add_patch(Rectangle((x, y), lado, lado, fill=False, ec=cor, lw=2))
            origem.set_title(f"Janela de {lado} × {lado} pixels", fontsize=11)
            destino.imshow(imagem[y:y + lado, x:x + lado], interpolation="nearest")
            destino.set_title(titulo, fontsize=11)
            for margem in destino.spines.values():
                margem.set_color(cor)
                margem.set_linewidth(2)
            origem.set_axis_off()
            destino.set_xticks([])
            destino.set_yticks([])
        fig.suptitle("Janelas definidas manualmente — sem inferência", fontsize=13)
        fig.supxlabel("Os recortes são exibidos com o mesmo tamanho visual.", fontsize=10)
    plt.close(fig)
    return fig


def detalhar(imagem, caixa, classe, score):
    """Relaciona uma previsão recebida aos seus quatro limites na imagem."""
    imagem = _rgb(imagem).permute(1, 2, 0).numpy()
    caixas = _caixas(caixa)
    if len(caixas) != 1:
        raise ValueError("Selecione exatamente uma caixa para detalhar.")
    x1, y1, x2, y2 = caixas[0].tolist()
    altura, largura = imagem.shape[:2]
    score = float(score)
    valores = [
        ["x₁ · esquerda", f"{x1:.2f}"],
        ["y₁ · superior", f"{y1:.2f}"],
        ["x₂ · direita", f"{x2:.2f}"],
        ["y₂ · inferior", f"{y2:.2f}"],
        ["Classe", str(classe)],
        ["Score", f"{score:.4f}"],
    ]

    with rc_context(ESTILO):
        fig, eixos = plt.subplots(
            1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [2.1, 1]},
            layout="constrained",
        )
        foto, dados = eixos
        foto.imshow(imagem, extent=(0, largura, altura, 0))
        foto.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COR_CAIXA, lw=2))
        foto.scatter([x1, x2], [y1, y2], s=35, color=COR_CAIXA, edgecolor="#202830", zorder=3)
        for x, y, texto, deslocamento in [
            (x1, y1, "(x₁, y₁)", (6, 8)), (x2, y2, "(x₂, y₂)", (6, -16)),
        ]:
            foto.annotate(
                texto, (x, y), xytext=deslocamento, textcoords="offset points",
                fontsize=10, color="white", bbox={"facecolor": "#202830", "edgecolor": "none", "alpha": 0.85},
            )
        foto.xaxis.tick_top()
        foto.xaxis.set_label_position("top")
        foto.set_xlabel("x (pixels) →", labelpad=10)
        foto.set_ylabel("y (pixels)")
        foto.annotate(
            "", xy=(-0.075, 0.05), xytext=(-0.075, 0.22),
            xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#202830"},
            annotation_clip=False,
        )
        foto.set_xlim(0, largura)
        foto.set_ylim(altura, 0)
        foto.set_xticks([0, largura // 4, largura // 2, 3 * largura // 4, largura])
        foto.set_yticks([0, altura // 4, altura // 2, 3 * altura // 4, altura])
        dados.set_axis_off()
        dados.set_title("Valores da mesma previsão", fontsize=12)
        quadro = dados.table(cellText=valores, colLabels=["Campo", "Valor"], loc="center", cellLoc="left")
        quadro.auto_set_font_size(False)
        quadro.set_fontsize(11)
        quadro.scale(1, 2.1)
        for (linha, coluna), celula in quadro.get_celld().items():
            celula.set_edgecolor("#dce1e6")
            celula.set_facecolor("#e9eef3" if linha == 0 else "white")
            if linha == 0:
                celula.set_text_props(weight="bold")
        dados.text(0.5, 0.04, "Formato xyxy · coordenadas em pixels", ha="center", fontsize=10, transform=dados.transAxes)
    plt.close(fig)
    return fig


def tabela(caixas, classes, scores):
    """Exibe até 20 previsões na ordem recebida; a seleção pertence ao notebook."""
    caixas = _caixas(caixas)
    scores = torch.as_tensor(scores).detach().cpu().reshape(-1)
    classes = list(classes)
    if len(caixas) != len(classes) or len(caixas) != len(scores):
        raise ValueError("caixas, classes e scores devem conter o mesmo número de previsões.")
    if len(caixas) > 20:
        raise ValueError("Selecione explicitamente até 20 previsões antes de montar a tabela.")
    estilo = 'style="padding:8px 12px;border-bottom:1px solid #dce1e6;text-align:left"'
    colunas = ["Índice na seleção", "x₁", "y₁", "x₂", "y₂", "Classe", "Score"]
    cabecalho = "".join(f"<th {estilo}>{coluna}</th>" for coluna in colunas)
    linhas = []
    for indice, (caixa, classe, score) in enumerate(zip(caixas, classes, scores)):
        valores = [str(indice), *(f"{valor:.2f}" for valor in caixa), escape(str(classe)), f"{score:.4f}"]
        linhas.append("<tr>" + "".join(f"<td {estilo}>{valor}</td>" for valor in valores) + "</tr>")
    if not linhas:
        linhas.append(f'<tr><td colspan="7" {estilo}>Nenhuma detecção recebida.</td></tr>')
    return HTML(
        '<table style="font-family:DejaVu Sans,Arial,sans-serif;border-collapse:collapse">'
        f"<thead><tr>{cabecalho}</tr></thead><tbody>{''.join(linhas)}</tbody></table>"
    )
