"""Extração verificada da reserva Other_farms, somente após congelar a receita.

Não executa modelos. Other_farms é uma pasta agregada: não estabelece identidade
de uma fazenda nem independência geográfica em relação ao conjunto principal.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .experimento import ler_caixas
from .preparar_dados import sha256

ROOT = Path(__file__).resolve().parents[1]


def obter_reserva(destino: str | Path | None = None) -> Path:
    """Valida e extrai 62 imagens/test + labels/test; retorna dados/reserva.

    Deve ser chamado somente depois de congelar pesos, limiar, resolução,
    recortes e deduplicação usando o conjunto principal e sua validação.
    """
    assets = ROOT/'assets'
    index = json.loads((assets/'reserva-pacotes.json').read_text())
    raw_manifest = (assets/'reserva-manifesto.json').read_bytes()
    if sha256(raw_manifest) != index['manifest_sha256']:
        raise RuntimeError('Manifesto da reserva não corresponde aos pacotes.')
    manifest = json.loads(raw_manifest)
    if manifest['dataset'] != index['dataset'] or len(manifest['samples']) != 62:
        raise RuntimeError('Versão ou quantidade inesperada da reserva.')
    destination = Path(destino).resolve() if destino is not None else ROOT/'dados'/'reserva'
    destination.mkdir(parents=True, exist_ok=True)
    expected = {row[key] for row in manifest['samples'] for key in ('image','label')}
    existing = {p.relative_to(destination).as_posix() for name in ('images','labels')
                for p in (destination/name).rglob('*') if p.is_file()
                and p.suffix != '.cache' and p.name != '.DS_Store'}
    if existing - expected:
        raise RuntimeError('A pasta da reserva contém imagens/rótulos adicionais; use outra pasta.')
    for package in index['packages']:
        path = assets/package['filename']
        if path.stat().st_size != package['size'] or sha256(path.read_bytes()) != package['sha256']:
            raise RuntimeError(f'Pacote da reserva ausente ou corrompido: {path.name}')
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = (destination/info.filename).resolve()
                if destination not in target.parents or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise RuntimeError('Caminho inválido ou link simbólico no pacote da reserva.')
            archive.extractall(destination)
    total = 0
    for row in manifest['samples']:
        if row['split'] != 'test' or row['farm'] != 'Other_farms':
            raise RuntimeError('Reserva com partição ou origem inesperada.')
        for path_key, hash_key in [('image','derived_sha256'),('label','label_sha256')]:
            if sha256((destination/row[path_key]).read_bytes()) != row[hash_key]:
                raise RuntimeError(f'Arquivo da reserva corrompido: {row[path_key]}')
        with Image.open(destination/row['image']) as image:
            if image.size != (row['width'],row['height']) or image.format != 'JPEG' or image.mode != 'RGB':
                raise RuntimeError('Imagem da reserva com dimensões/formato diferentes do manifesto.')
        boxes = ler_caixas(destination/row['label'], row['width'], row['height'])
        if len(boxes) != row['count'] or len(boxes) == 0:
            raise RuntimeError('Contagem de referência da reserva divergente.')
        total += len(boxes)
    if total != 1355:
        raise RuntimeError('A reserva exige exatamente 1.355 caixas.')
    (destination/'manifesto.json').write_bytes(raw_manifest)
    (destination/'data.yaml').write_text(f'path: {json.dumps(str(destination))}\ntest: images/test\nnames:\n  0: cow\n')
    return destination


def inventario_reserva(root: str | Path) -> pd.DataFrame:
    """Inventário compatível com experimento.inventario, contendo só split test.

    Não cria treino/validação artificiais e não calcula previsões de modelos.
    Caminhos absolutos; median_box_side usa a mesma mediana de sqrt(área em px).
    """
    root = Path(root).resolve()
    manifest = json.loads((root/'manifesto.json').read_text())
    rows = []
    for sample in sorted(manifest['samples'],key=lambda r:r['image']):
        image_path, label_path = root/sample['image'], root/sample['label']
        with Image.open(image_path) as image:
            width,height=image.size
        boxes=ler_caixas(label_path,width,height)
        side=np.sqrt(np.prod(boxes[:,2:]-boxes[:,:2],axis=1))
        rows.append({'image_id':f'test/{image_path.name}','split':'test',
                     'image':str(image_path),'label':str(label_path),
                     'width':width,'height':height,'count':len(boxes),
                     'median_box_side':float(np.median(side)) if len(side) else np.nan})
    return pd.DataFrame(rows,columns=['image_id','split','image','label','width','height','count','median_box_side'])
