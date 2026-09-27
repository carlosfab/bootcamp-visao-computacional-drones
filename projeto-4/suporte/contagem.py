"""Detecção e contagem de gado, com métricas auditáveis e operações em NumPy.

Caixas usam pixels ``[x_min, y_min, x_max, y_max]``, sem ``+1`` na área.
Os helpers de inferência recebem imagens RGB uint8 (array, PIL ou caminho),
convertem para o BGR esperado por Ultralytics e retornam a classe única 0.
Não é necessário importar Ultralytics ou PyTorch para calcular as métricas.
"""

from pathlib import Path

import numpy as np


def _boxes(value):
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("Caixas precisam ter formato (N, 4).")
    if not np.isfinite(array).all() or (array[:, 2:] < array[:, :2]).any():
        raise ValueError("Caixas precisam ser finitas, com máximos >= mínimos.")
    return array


def _scores(value, size):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("Scores precisam ser um vetor finito de tamanho N.")
    return array


def _threshold(value, name, allow_zero=True):
    if not np.isfinite(value) or not 0 <= value <= 1 or (not allow_zero and value == 0):
        raise ValueError(f"{name} precisa estar entre {'0' if allow_zero else '>0'} e 1.")


def iou_matrix(boxes1, boxes2):
    """Matriz (N, M) de IoU; caixas de área zero têm IoU zero."""
    first, second = _boxes(boxes1), _boxes(boxes2)
    intersection_sides = np.maximum(
        0, np.minimum(first[:, None, 2:], second[None, :, 2:])
        - np.maximum(first[:, None, :2], second[None, :, :2]),
    )
    intersection = intersection_sides.prod(axis=2)
    first_area = (first[:, 2:] - first[:, :2]).prod(axis=1)
    second_area = (second[:, 2:] - second[:, :2]).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_detections(gt_boxes, pred_boxes, scores, iou_threshold=0.5):
    """Associa previsões por score decrescente a GTs ainda livres (1:1).

    Empates de score preservam a ordem original; empates de IoU escolhem o
    menor índice de GT. ``matches`` contém índices originais e o IoU de cada
    associação. Não calcula AP: avalia um único ponto de operação.
    """
    _threshold(iou_threshold, "iou_threshold", allow_zero=False)
    gt, pred = _boxes(gt_boxes), _boxes(pred_boxes)
    confidence = _scores(scores, len(pred))
    overlaps = iou_matrix(pred, gt)
    available = np.ones(len(gt), dtype=bool)
    tp_indices, fp_indices, matches = [], [], []
    for pred_index in np.argsort(-confidence, kind="stable"):
        eligible = np.flatnonzero(available & (overlaps[pred_index] >= iou_threshold))
        if len(eligible):
            gt_index = int(eligible[np.argmax(overlaps[pred_index, eligible])])
            available[gt_index] = False
            tp_indices.append(int(pred_index))
            matches.append({"gt_index": gt_index, "pred_index": int(pred_index),
                            "iou": float(overlaps[pred_index, gt_index])})
        else:
            fp_indices.append(int(pred_index))
    fn_indices = np.flatnonzero(available).tolist()
    return {"tp": len(tp_indices), "fp": len(fp_indices), "fn": len(fn_indices),
            "tp_indices": tp_indices, "fp_indices": fp_indices,
            "fn_indices": fn_indices, "matches": matches}


def count_metrics(gt_counts, pred_counts):
    """Erros por imagem; bias positivo significa supercontagem.

    Acurácia exata é a fração de imagens cuja contagem coincide. Um conjunto
    vazio retorna ``None`` para médias/frações, pois não houve avaliação.
    """
    gt = np.asarray(gt_counts, dtype=np.float64)
    pred = np.asarray(pred_counts, dtype=np.float64)
    if gt.ndim != 1 or pred.shape != gt.shape:
        raise ValueError("Contagens precisam ser vetores com o mesmo tamanho.")
    for values in (gt, pred):
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Contagens precisam ser inteiros finitos e não negativos.")
    if len(gt) == 0:
        return {"n_images": 0, "mae": None, "rmse": None, "bias": None,
                "exact_accuracy": None, "exact_match_count": 0}
    errors = pred - gt
    return {"n_images": len(gt), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "bias": float(errors.mean()), "exact_accuracy": float((errors == 0).mean()),
            "exact_match_count": int((errors == 0).sum())}


def nms(boxes, scores, iou_threshold=0.5):
    """Índices mantidos por NMS; suprime IoU > limiar, com empates estáveis."""
    _threshold(iou_threshold, "iou_threshold")
    coordinates = _boxes(boxes)
    confidence = _scores(scores, len(coordinates))
    remaining = np.argsort(-confidence, kind="stable")
    kept = []
    while len(remaining):
        index = int(remaining[0])
        kept.append(index)
        rest = remaining[1:]
        overlap = iou_matrix(coordinates[index:index + 1], coordinates[rest])[0]
        remaining = rest[overlap <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def tile_windows(width, height, tile_size=640, overlap=0.2):
    """Janelas (x1,y1,x2,y2), em ordem de linhas, cobrindo toda a imagem.

    A última janela é encostada à borda; a sobreposição final pode ser maior
    que a solicitada. Imagens menores que um tile geram uma janela menor.
    """
    for name, value in (("width", width), ("height", height), ("tile_size", tile_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} precisa ser um inteiro positivo.")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap precisa estar em [0, 1).")
    step = max(1, int(round(tile_size * (1 - overlap))))

    def starts(length):
        last = max(0, length - tile_size)
        positions = list(range(0, last + 1, step))
        if positions[-1] != last:
            positions.append(last)
        return positions

    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in starts(height) for x in starts(width)]


def _image_rgb(image):
    if isinstance(image, (str, Path)):
        from PIL import Image
        with Image.open(image) as opened:
            array = np.asarray(opened.convert("RGB"))
    elif hasattr(image, "convert"):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or not all(array.shape[:2]) or array.dtype != np.uint8:
        raise ValueError("A imagem precisa ser RGB uint8, com formato (altura, largura, 3).")
    return array


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _prediction(boxes, scores, width, height):
    boxes = _boxes(boxes).copy()
    scores = _scores(scores, len(boxes))
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    nonempty = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    return {"boxes": boxes[nonempty], "scores": scores[nonempty],
            "class_ids": np.zeros(int(nonempty.sum()), dtype=np.int64)}


def predict_image(model, image, conf=0.05, imgsz=640, class_name="cow", device="cpu"):
    """YOLO numa imagem RGB; mantém cow/cattle por nome e remapeia para 0.

    ``model`` é um objeto YOLO já carregado. Outros nomes podem ser passados
    explicitamente. O retorno contém arrays ``boxes``, ``scores`` e
    ``class_ids``; caixas são limitadas às bordas e áreas vazias descartadas.
    ``device='cpu'`` é o padrão; passe ``device=0`` para usar a GPU disponível.
    """
    _threshold(conf, "conf")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name precisa ser um nome de classe não vazio.")
    rgb = _image_rgb(image)
    result = model.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]), conf=conf,
                           imgsz=imgsz, device=device, verbose=False)[0]
    target = class_name.strip().lower()
    accepted = {"cow", "cattle"} if target in {"cow", "cattle"} else {target}
    names = getattr(result, "names", None)
    if names is None:
        names = model.names
    name_items = names.items() if hasattr(names, "items") else enumerate(names)
    allowed_ids = {int(index) for index, name in name_items if str(name).strip().lower() in accepted}
    if not allowed_ids:
        raise ValueError(f"O modelo não tem a classe solicitada: {sorted(accepted)}.")
    if result.boxes is None:
        return _prediction([], [], rgb.shape[1], rgb.shape[0])
    boxes = _boxes(_numpy(result.boxes.xyxy))
    scores = _scores(_numpy(result.boxes.conf), len(boxes))
    classes = _numpy(result.boxes.cls).reshape(-1)
    if len(classes) != len(boxes):
        raise ValueError("O modelo retornou números incompatíveis de caixas e classes.")
    selected = np.isin(classes, list(allowed_ids)) & (scores >= conf)
    return _prediction(boxes[selected], scores[selected], rgb.shape[1], rgb.shape[0])


def predict_tiled(model, image, conf=0.05, imgsz=640, class_name="cow",
                  tile_size=640, overlap=0.2, nms_iou=0.5, device="cpu"):
    """Inferência sequencial em tiles, coordenadas globais e NMS final.

    NMS reduz duplicatas sobrepostas, mas não garante união de animais
    fragmentados nas bordas. Avalie esse erro visualmente no conjunto válido.
    Não some contagens dos tiles: conte as caixas retornadas após NMS.
    """
    _threshold(nms_iou, "nms_iou")
    rgb = _image_rgb(image)
    all_boxes, all_scores = [], []
    for x1, y1, x2, y2 in tile_windows(rgb.shape[1], rgb.shape[0], tile_size, overlap):
        prediction = predict_image(model, rgb[y1:y2, x1:x2], conf, imgsz, class_name, device)
        all_boxes.append(prediction["boxes"] + np.asarray([x1, y1, x1, y1]))
        all_scores.append(prediction["scores"])
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores)
    kept = nms(boxes, scores, nms_iou)
    return _prediction(boxes[kept], scores[kept], rgb.shape[1], rgb.shape[0])


def _detection_metrics(tp, fp, fn):
    # Convenção explícita: uma razão sem denominador vale 0, inclusive vazio/vazio.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(rows, conf_threshold=0.25, iou_threshold=0.5):
    """Avalia linhas com image_id, gt_boxes, pred_boxes e scores (JSON-ready).

    Reutilize previsões feitas com confiança baixa ao calibrar o limiar na
    validação; não escolha limiares olhando o teste. IDs repetidos são erro.
    ``summary`` agrega TP/FP/FN (micro), contagens e os limiares usados;
    ``per_image`` permite localizar falsos positivos/negativos e erros.
    """
    _threshold(conf_threshold, "conf_threshold")
    _threshold(iou_threshold, "iou_threshold", allow_zero=False)
    per_image, seen = [], set()
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in seen:
            raise ValueError(f"image_id repetido: {image_id}")
        seen.add(image_id)
        gt, pred = _boxes(row["gt_boxes"]), _boxes(row["pred_boxes"])
        scores = _scores(row["scores"], len(pred))
        selected = scores >= conf_threshold
        matched = match_detections(gt, pred[selected], scores[selected], iou_threshold)
        gt_count, pred_count = len(gt), int(selected.sum())
        per_image.append({"image_id": image_id, "gt_count": gt_count,
                          "pred_count": pred_count, "count_error": pred_count - gt_count,
                          **_detection_metrics(matched["tp"], matched["fp"], matched["fn"])})
    totals = {key: sum(row[key] for row in per_image) for key in ("tp", "fp", "fn")}
    summary = {**_detection_metrics(**totals),
               **count_metrics([r["gt_count"] for r in per_image], [r["pred_count"] for r in per_image]),
               "conf_threshold": float(conf_threshold), "iou_threshold": float(iou_threshold)}
    return {"summary": summary, "per_image": per_image}
