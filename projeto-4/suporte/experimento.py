"""Rotina reproduzível de avaliação: inventário, inferência, calibração e figuras.

O dataset deve conter images/{train,val,test} e labels/{train,val,test}, com
anotações YOLO de uma classe (0 = cow). Label vazio representa imagem negativa;
label ausente é erro. Não há downloads nem alterações nos dados de entrada.

As métricas de detecção usam associação 1:1 com IoU >= 0,5 em um único limiar
de confiança. Não são AP/mAP. A confiança é escolhida somente na validação e
depois mantida fixa no teste. A contagem descreve cada imagem; não identifica
animais únicos entre fotografias ou frames sobrepostos.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

from .contagem import evaluate_predictions, predict_image, predict_tiled


THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ler_caixas(label, width, height):
    """Lê YOLO ``0 cx cy w h`` normalizado e retorna caixas Nx4 em pixels.

    Valida classe única, cinco campos, números finitos, dimensões positivas e
    caixas dentro da imagem. Tolera apenas 1e-6 de arredondamento nas bordas
    normalizadas, limitando essa diferença à imagem. Labels ausentes não são
    interpretados silenciosamente como imagens negativas.
    """
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} precisa ser um inteiro positivo.")
    path = Path(label)
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        prefix = f"{path.name}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"{prefix}: esperado '0 cx cy w h' (cinco campos).")
        try:
            values = np.asarray([float(value) for value in fields])
        except ValueError as exc:
            raise ValueError(f"{prefix}: campo não numérico.") from exc
        if not np.isfinite(values).all() or values[0] != 0:
            raise ValueError(f"{prefix}: valores devem ser finitos e classe deve ser 0 (cow).")
        cx, cy, box_width, box_height = values[1:]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            raise ValueError(f"{prefix}: coordenadas YOLO inválidas ou não normalizadas.")
        box = np.asarray([cx - box_width / 2, cy - box_height / 2,
                          cx + box_width / 2, cy + box_height / 2])
        if (box < -1e-6).any() or (box > 1 + 1e-6).any():
            raise ValueError(f"{prefix}: caixa ultrapassa os limites da imagem.")
        boxes.append(np.clip(box, 0, 1) * [width, height, width, height])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def inventario(root):
    """DataFrame por imagem, ordenado por split e caminho relativo.

    Colunas: image_id, split, image, label, width, height, count,
    median_box_side. Caminhos são absolutos. ``median_box_side`` é a mediana
    de sqrt(área da caixa), em pixels, e NaN para uma imagem sem animais.
    image_id inclui o split e a extensão para impedir colisões de nomes.
    """
    root = Path(root).resolve()
    rows = []
    for split in ("train", "val", "test"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Split incompleto: {image_dir} e {label_dir} são necessários.")
        images = sorted(path for path in image_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
        labels_seen = set()
        for image_path in images:
            relative = image_path.relative_to(image_dir)
            label_path = label_dir / relative.with_suffix(".txt")
            if label_path in labels_seen:
                raise ValueError(f"Duas imagens compartilham o mesmo label: {label_path}")
            labels_seen.add(label_path)
            with Image.open(image_path) as opened:
                width, height = opened.size
            boxes = ler_caixas(label_path, width, height)
            side = np.sqrt(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))
            rows.append({"image_id": f"{split}/{relative.as_posix()}", "split": split,
                         "image": str(image_path), "label": str(label_path),
                         "width": width, "height": height, "count": len(boxes),
                         "median_box_side": float(np.median(side)) if len(side) else np.nan})
    return pd.DataFrame(rows, columns=["image_id", "split", "image", "label", "width", "height",
                                       "count", "median_box_side"])


def _rgb(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return np.asarray(opened.convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8 or not all(array.shape[:2]):
        raise ValueError("A imagem precisa ser RGB uint8 com formato (altura, largura, 3).")
    return array


def inferir(model, table, device="cpu", tiled=False):
    """Infere uma tabela de inventário sequencialmente, com conf=0,05.

    Retorna lista JSON: image_id, split, image, gt_boxes, pred_boxes, scores,
    seconds. O tempo inclui a chamada ao modelo e o pós-processamento, exclui
    leitura da imagem/label; a primeira chamada pode incluir aquecimento.
    Tiles usam tamanho 640, overlap 0,2 e NMS global IoU 0,5. Nenhuma contagem
    de tile é somada. ``device=0`` permite a GPU; CPU é o padrão.
    """
    required = {"image_id", "split", "image", "label", "width", "height"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no inventário: {sorted(missing)}")
    if table["image_id"].duplicated().any():
        raise ValueError("O inventário contém image_id repetido.")
    predictor = predict_tiled if tiled else predict_image
    rows = []
    for row in table.to_dict("records"):
        rgb = _rgb(row["image"])
        height, width = rgb.shape[:2]
        if (width, height) != (row["width"], row["height"]):
            raise ValueError(f"Dimensões mudaram desde o inventário: {row['image_id']}")
        gt = ler_caixas(row["label"], width, height)
        start = perf_counter()
        prediction = predictor(model, rgb, conf=THRESHOLDS[0], device=device)
        seconds = perf_counter() - start
        rows.append({"image_id": str(row["image_id"]), "split": str(row["split"]),
                     "image": str(row["image"]), "gt_boxes": gt.tolist(),
                     "pred_boxes": prediction["boxes"].tolist(),
                     "scores": prediction["scores"].tolist(), "seconds": seconds})
    return rows


def calibrar(rows):
    """Retorna (limiar, DataFrame) usando somente imagens de validação.

    Maximiza F1 micro; empates são resolvidos pelo menor MAE e então pelo
    maior limiar. A tabela mantém ordem crescente do limiar, coluna
    ``threshold``. Cada linha deve declarar ``split='val'``. Não calcula
    AP/mAP e não modifica as previsões brutas.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("É necessário ao menos uma imagem para calibrar.")
    if any(row.get("split") != "val" for row in rows):
        raise ValueError("Calibre somente no split val; mantenha o teste intocado.")
    results = []
    for threshold in THRESHOLDS:
        summary = evaluate_predictions(rows, conf_threshold=threshold)["summary"]
        summary["threshold"] = summary.pop("conf_threshold")
        results.append(summary)
    table = pd.DataFrame(results)
    best = table.sort_values(["f1", "mae", "threshold"], ascending=[False, True, False],
                             kind="stable").iloc[0]
    return float(best["threshold"]), table


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Valor não serializável: {type(value).__name__}")


def salvar_resultado(path, rows, threshold):
    """Salva predictions.json, per_image.csv e resumo.json em um diretório.

    Retorna o mesmo dict gravado em resumo.json: métricas gerais no primeiro
    nível, ``by_presence`` com positive/negative (definidos pelo GT) e
    ``timing``. O subconjunto negativo inclui false_positive_image_rate.
    Subconjuntos vazios têm médias de contagem None; razões de detecção sem
    denominador seguem a convenção zero de contagem.py. Tempos não medidos
    ficam ausentes da média (n_timed_images torna isso explícito).
    """
    rows = list(rows)
    evaluation = evaluate_predictions(rows, conf_threshold=threshold)
    summary = dict(evaluation["summary"])
    by_presence = {}
    for name, positive in (("positive", True), ("negative", False)):
        subset = [row for row in rows if bool(len(row["gt_boxes"])) == positive]
        result = evaluate_predictions(subset, conf_threshold=threshold)
        by_presence[name] = result["summary"]
        if not positive:
            by_presence[name]["false_positive_image_rate"] = (
                sum(row["pred_count"] > 0 for row in result["per_image"]) / len(subset)
                if subset else None)
    times = [float(row["seconds"]) for row in rows if "seconds" in row]
    if times and (not np.isfinite(times).all() or min(times) < 0):
        raise ValueError("Tempos de inferência precisam ser finitos e não negativos.")
    summary["by_presence"] = by_presence
    summary["timing"] = {"n_timed_images": len(times),
                         "total_seconds": float(sum(times)) if times else None,
                         "mean_seconds": float(np.mean(times)) if times else None}
    summary["metric_scope"] = "Detecção em ponto de operação, IoU >= 0.5; não é AP/mAP."
    extra = {str(row["image_id"]): row for row in rows}
    per_image = []
    for row in evaluation["per_image"]:
        source = extra[row["image_id"]]
        per_image.append({**row, **{key: source[key] for key in ("split", "image", "seconds")
                                   if key in source}})
    columns = ["image_id", "gt_count", "pred_count", "count_error", "tp", "fp", "fn",
               "precision", "recall", "f1", "split", "image", "seconds"]
    # Serializar antes de criar arquivos para detectar NaN/objetos inválidos cedo.
    predictions_text = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=_json_default) + "\n"
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False,
                              default=_json_default) + "\n"
    per_image_text = pd.DataFrame(per_image, columns=columns).to_csv(index=False)
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "predictions.json").write_text(predictions_text, encoding="utf-8")
    (directory / "per_image.csv").write_text(per_image_text, encoding="utf-8")
    (directory / "resumo.json").write_text(summary_text, encoding="utf-8")
    return summary


def desenhar(image, gt_boxes, pred_boxes, scores, threshold):
    """Figura Matplotlib: GT verde, predição vermelha e confiança por caixa.

    Aceita imagem RGB uint8, PIL ou caminho. Retorna Figure; use plt.show()
    no notebook e plt.close(fig) após salvar várias figuras. Não modifica a
    imagem original. A tipografia é sans-serif (DejaVu Sans).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    rgb = _rgb(image)
    # Reutiliza a validação e a definição do limiar das métricas.
    row = {"image_id": "figura", "gt_boxes": gt_boxes, "pred_boxes": pred_boxes, "scores": scores}
    evaluated = evaluate_predictions([row], conf_threshold=threshold)["per_image"][0]
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    predictions = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    confidence = np.asarray(scores, dtype=float)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig, axis = plt.subplots(figsize=(12, 8), layout="constrained")
        axis.imshow(rgb)
        for x1, y1, x2, y2 in gt:
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                     edgecolor="#20c55a", linewidth=2))
        for (x1, y1, x2, y2), score in zip(predictions, confidence):
            if score < threshold:
                continue
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                     edgecolor="#ef4444", linewidth=1.5, linestyle="--"))
            axis.text(x1, max(0, y1 - 2), f"{score:.2f}", color="white", fontsize=8,
                      bbox={"facecolor": "#b91c1c", "alpha": .8, "edgecolor": "none", "pad": 1})
        axis.set_title(f"Referência: {evaluated['gt_count']} animais | "
                       f"Predição: {evaluated['pred_count']} | Confiança ≥ {threshold:.2f}")
        axis.legend(handles=[Patch(facecolor="none", edgecolor="#20c55a", label="Referência (GT)"),
                             Patch(facecolor="none", edgecolor="#ef4444", linestyle="--",
                                   label="Predição")], loc="upper right")
        axis.set_xlim(0, rgb.shape[1])
        axis.set_ylim(rgb.shape[0], 0)
        axis.axis("off")
    return fig
