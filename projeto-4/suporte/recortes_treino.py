"""Recortes de treino/validação, sem ler imagens ou labels do teste.

``preparar_recortes(raiz_base)`` retorna ``raiz_base.parent / 'gado-recortes'``.
A origem é o dataset com manifesto produzido por ``obter_dados()``. Fotografias,
labels e partições de origem nunca são alterados. Não há downloads ou treino.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

from .contagem import tile_windows
from .experimento import ler_caixas


_DATASET = "gado-recortes-v1"
_SPLITS = ("train", "val")
_CONFIG = {
    "schema_version": 1,
    "dataset": _DATASET,
    "tile_size": 640,
    "overlap": 0.2,
    "min_visibility": 0.5,
    "min_clipped_side_px": 2.0,
    "ambiguous_tile_policy": "discard_entire_tile",
    "train_negative_ratio": 1,
    "negative_seed": _DATASET,
    "val_negative_sampling": "none",
    "jpeg_quality": 95,
    "jpeg_subsampling": 0,
    "resize": False,
    "pillow_version": pillow_version,
}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _source_path(base, relative, kind, split):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or name.parts[:2] != (kind, split):
        raise ValueError(f"Caminho de origem fora de {kind}/{split}: {relative}")
    result = base / relative
    if not result.resolve().is_relative_to(base / kind / split):
        raise ValueError(f"Caminho de origem redirecionado para outra partição: {relative}")
    return result


def _validar_origem(base, manifest):
    """Confere somente arquivos train/val e devolve registros ordenados."""
    parents = []
    seen = set()
    groups = {}
    for sample in manifest["samples"]:
        if sample["split"] not in _SPLITS:
            continue
        split, parent_id = sample["split"], sample["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", parent_id) or parent_id in seen:
            raise ValueError(f"ID de origem inválido ou repetido: {parent_id}")
        seen.add(parent_id)
        image = _source_path(base, sample["image"], "images", split)
        label = _source_path(base, sample["label"], "labels", split)
        if _digest(image.read_bytes()) != sample["derived_sha256"]:
            raise ValueError(f"Hash da imagem de origem divergente: {sample['image']}")
        if _digest(label.read_bytes()) != sample["label_sha256"]:
            raise ValueError(f"Hash do label de origem divergente: {sample['label']}")
        for key in ("flight", "date"):
            if sample.get("farm") and sample.get(key):
                group = (key, sample["farm"], sample[key])
                if group in groups and groups[group] != split:
                    raise ValueError("Grupo de captura de origem atravessa treino e validação.")
                groups[group] = split
        parents.append(sample)
    return sorted(parents, key=lambda row: (_SPLITS.index(row["split"]), row["id"]))


def _caixas_do_recorte(boxes, window):
    """Caixas locais; None invalida o tile inteiro por fragmentação ambígua."""
    x1, y1, x2, y2 = window
    clipped = boxes.copy()
    clipped[:, :2] = np.maximum(clipped[:, :2], [x1, y1])
    clipped[:, 2:] = np.minimum(clipped[:, 2:], [x2, y2])
    sides = clipped[:, 2:] - clipped[:, :2]
    intersects = (sides > 0).all(axis=1)
    if not intersects.any():
        return np.empty((0, 4), dtype=np.float64)
    clipped, sides = clipped[intersects], sides[intersects]
    areas = np.prod(boxes[intersects, 2:] - boxes[intersects, :2], axis=1)
    visibility = np.prod(sides, axis=1) / areas
    if ((visibility < _CONFIG["min_visibility"]).any()
            or (sides < _CONFIG["min_clipped_side_px"]).any()):
        return None
    return clipped - [x1, y1, x1, y1]


def _label_bytes(boxes, width, height):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2 / [width, height]
    sides = (boxes[:, 2:] - boxes[:, :2]) / [width, height]
    return "".join("0 " + " ".join(f"{value:.10f}" for value in row) + "\n"
                   for row in np.column_stack([centers, sides])).encode("utf-8")


def _yaml(destination):
    return (f"path: {json.dumps(str(destination))}\ntrain: images/train\n"
            "val: images/val\nnames:\n  0: cow\n").encode("utf-8")


def _verificar_existente(destination, source_hash, config_bytes):
    """Reuso exige identidade e integridade; não corrige/sobrescreve arquivos."""
    try:
        config_existing = (destination / "configuracao.json").read_bytes()
        manifest = json.loads((destination / "manifesto.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Destino existente sem manifesto/configuração válidos; use outra pasta-base.") from exc
    if (config_existing != config_bytes or manifest.get("dataset") != _DATASET
            or manifest.get("config_sha256") != _digest(config_bytes)
            or manifest.get("source_manifest_sha256") != source_hash
            or manifest.get("configuration") != _CONFIG):
        raise ValueError("Destino pertence a outra origem/configuração; nenhum arquivo foi sobrescrito.")
    expected = set()
    for sample in manifest["samples"]:
        if sample["split"] not in _SPLITS:
            raise ValueError("O destino contém uma partição não autorizada.")
        for kind, hash_key in (("image", "image_sha256"), ("label", "label_sha256")):
            relative = sample[kind]
            path = _source_path(destination, relative,
                                "images" if kind == "image" else "labels", sample["split"])
            if relative in expected or _digest(path.read_bytes()) != sample[hash_key]:
                raise ValueError(f"Recorte ausente, duplicado ou alterado: {relative}")
            expected.add(relative)
    # Ultralytics pode criar estes caches durante um treino posterior.
    cache_paths = {f"labels/{split}.cache" for split in _SPLITS}
    actual = {p.relative_to(destination).as_posix() for folder in ("images", "labels")
              for p in (destination / folder).rglob("*") if p.is_file()}
    if actual - cache_paths != expected:
        raise ValueError("Arquivos de imagens/labels do destino não correspondem ao manifesto.")
    if (destination / "data.yaml").read_bytes() != _yaml(destination):
        raise ValueError("data.yaml do destino foi alterado; nenhum arquivo foi sobrescrito.")


def preparar_recortes(raiz_base: Path) -> Path:
    """Prepara ou verifica ``dados/gado-recortes``, usando somente train/val.

    Tiles: 640 px, overlap 0,2, JPEG qualidade 95 e sem redimensionar. Um tile
    inteiro é descartado se qualquer caixa intersectante retiver menos de 50%
    da área original ou ficar com lado menor que 2 px. Os demais mantêm todas
    as caixas intersectantes após clipping. Treino: todos os positivos e até
    um negativo por positivo, escolhidos por SHA-256 com seed gado-recortes-v1.
    Validação: todos os tiles válidos, inclusive negativos. Não existe test.

    A configuração, o manifesto de origem e todos os arquivos train/val são
    conferidos antes de reutilizar uma preparação. Divergência causa erro,
    sem sobrescrever um dataset anterior. Um build novo é publicado somente
    após terminar; falhas deixam a origem e um eventual destino preservados.
    """
    base = Path(raiz_base).resolve()
    destination = base.parent / "gado-recortes"
    if destination == base:
        raise ValueError("A origem não pode ser o próprio diretório gado-recortes.")
    source_bytes = (base / "manifesto.json").read_bytes()
    source_hash, config_bytes = _digest(source_bytes), _json_bytes(_CONFIG)
    source = json.loads(source_bytes)
    if destination.exists():
        _verificar_existente(destination, source_hash, config_bytes)
        _validar_origem(base, source)
        return destination
    parents = _validar_origem(base, source)
    candidates = []
    summary = {split: {"source_images": 0, "candidate_tiles": 0,
                       "discarded_ambiguous_tiles": 0, "discarded_negative_tiles": 0,
                       "images": 0, "positive": 0, "negative": 0, "boxes": 0}
               for split in _SPLITS}
    for parent in parents:
        split = parent["split"]
        with Image.open(base / parent["image"]) as image:
            width, height = image.size
        if (width, height) != (parent["width"], parent["height"]):
            raise ValueError(f"Dimensões da origem divergentes: {parent['id']}")
        boxes = ler_caixas(base / parent["label"], width, height)
        summary[split]["source_images"] += 1
        for window in tile_windows(width, height, _CONFIG["tile_size"], _CONFIG["overlap"]):
            summary[split]["candidate_tiles"] += 1
            cropped = _caixas_do_recorte(boxes, window)
            if cropped is None:
                summary[split]["discarded_ambiguous_tiles"] += 1
                continue
            tile_id = f"{parent['id']}__x{window[0]:05d}_y{window[1]:05d}"
            candidates.append({"id": tile_id, "parent": parent, "window": window,
                               "boxes": cropped})
    negatives = [row for row in candidates
                 if row["parent"]["split"] == "train" and not len(row["boxes"])]
    n_positive = sum(row["parent"]["split"] == "train" and bool(len(row["boxes"]))
                     for row in candidates)
    negatives.sort(key=lambda row: (_digest(f"{_CONFIG['negative_seed']}\n{row['id']}".encode()), row["id"]))
    chosen_negatives = {row["id"] for row in negatives[:n_positive]}
    summary["train"]["discarded_negative_tiles"] = len(negatives) - len(chosen_negatives)
    selected = [row for row in candidates if row["parent"]["split"] == "val"
                or len(row["boxes"]) or row["id"] in chosen_negatives]
    records = []
    with tempfile.TemporaryDirectory(prefix=".gado-recortes-build-", dir=base.parent) as temporary:
        staged = Path(temporary) / "dataset"
        for split in _SPLITS:
            for kind in ("images", "labels"):
                (staged / kind / split).mkdir(parents=True)
        # Decodifica cada fotografia uma vez, preservando ordem determinística.
        current_parent, opened = None, None
        try:
            for row in selected:
                parent = row["parent"]
                if parent["id"] != current_parent:
                    if opened is not None:
                        opened.close()
                    with Image.open(base / parent["image"]) as source_image:
                        opened = source_image.convert("RGB")
                    current_parent = parent["id"]
                split, window = parent["split"], row["window"]
                width, height = window[2] - window[0], window[3] - window[1]
                image_relative = f"images/{split}/{row['id']}.jpg"
                label_relative = f"labels/{split}/{row['id']}.txt"
                with opened.crop(window) as crop:
                    crop.save(staged / image_relative, format="JPEG",
                              quality=_CONFIG["jpeg_quality"], subsampling=_CONFIG["jpeg_subsampling"])
                label_data = _label_bytes(row["boxes"], width, height)
                (staged / label_relative).write_bytes(label_data)
                records.append({"id": row["id"], "parent_id": parent["id"], "split": split,
                                "window": list(window), "width": width, "height": height,
                                "image": image_relative, "label": label_relative,
                                "count": len(row["boxes"]),
                                "image_sha256": _digest((staged / image_relative).read_bytes()),
                                "label_sha256": _digest(label_data),
                                "source_image": parent["image"], "source_label": parent["label"],
                                **{key: parent[key] for key in ("farm", "flight", "date", "original_path")
                                   if key in parent}})
                summary[split]["images"] += 1
                summary[split]["positive" if len(row["boxes"]) else "negative"] += 1
                summary[split]["boxes"] += len(row["boxes"])
        finally:
            if opened is not None:
                opened.close()
        manifest = {"schema_version": 1, "dataset": _DATASET,
                    "source_dataset": source["dataset"], "source_manifest_sha256": source_hash,
                    "configuration": _CONFIG, "config_sha256": _digest(config_bytes),
                    "summary": summary, "samples": records}
        (staged / "manifesto.json").write_bytes(_json_bytes(manifest))
        (staged / "configuracao.json").write_bytes(config_bytes)
        (staged / "data.yaml").write_bytes(_yaml(destination))
        # Reconfere a origem após o processamento, antes de publicar os recortes.
        if (base / "manifesto.json").read_bytes() != source_bytes:
            raise ValueError("O manifesto de origem mudou durante a preparação.")
        _validar_origem(base, source)
        if destination.exists():
            raise FileExistsError("O destino foi criado por outra execução; nenhum arquivo foi sobrescrito.")
        staged.rename(destination)
    return destination
