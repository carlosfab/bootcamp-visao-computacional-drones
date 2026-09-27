#!/usr/bin/env python3
"""Valida integridade da reserva e preservação do núcleo; não avalia modelos."""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from suporte.reserva import obter_reserva,inventario_reserva
from suporte.preparar_dados import sha256


def main():
    destination=obter_reserva()
    table=inventario_reserva(destination)
    if len(table)!=62 or int(table['count'].sum())!=1355 or set(table['split'])!={'test'}:
        raise RuntimeError('Inventário da reserva divergente.')
    if len(list((destination/'images/test').glob('*.jpg')))!=62 or len(list((destination/'labels/test').glob('*.txt')))!=62:
        raise RuntimeError('Quantidade real de arquivos da reserva divergente.')
    for path in table['image']:
        with Image.open(path) as image:image.load()
    base_manifest=json.loads((ROOT/'assets/manifesto.json').read_text())
    checked=0
    for row in base_manifest['samples']:
        for path_key,hash_key in [('image','derived_sha256'),('label','label_sha256')]:
            path=ROOT/'dados/gado'/row[path_key]
            if sha256(path.read_bytes())!=row[hash_key]:
                raise RuntimeError(f'Arquivo base diferente do manifesto: {row[path_key]}')
            checked+=1
    if checked!=600:raise RuntimeError('O núcleo deve ter600 arquivos de amostras.')
    index=json.loads((ROOT/'assets/reserva-pacotes.json').read_text())
    if any(p['size']>=40_000_000 for p in index['packages']):raise RuntimeError('Pacote excede40MB.')
    report={'status':'passed','scope':'Data integrity only. No visual inspection or model evaluation.',
            'images':62,'labels':62,'positive_images':62,'boxes':1355,'base_files_verified_unchanged':600,
            'manifest_sha256':sha256((ROOT/'assets/reserva-manifesto.json').read_bytes()),
            'base_manifest_sha256':sha256((ROOT/'assets/manifesto.json').read_bytes()),
            'packages':len(index['packages']),'package_total_bytes':index['total_size'],
            'largest_package_bytes':max(p['size'] for p in index['packages'])}
    (ROOT/'assets/reserva-validacao.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
