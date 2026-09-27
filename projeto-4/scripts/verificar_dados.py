#!/usr/bin/env python3
"""Confere pacotes, JPEGs, rótulos e grupos sem avaliar qualquer modelo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'suporte'))
from preparar_dados import obter_dados, sha256


def main():
    destination = obter_dados()
    manifest_path = ROOT/'assets'/'manifesto.json'
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['samples']
    counts = {}
    groups = {}
    for row in rows:
        with Image.open(destination/row['image']) as image:
            image.load()
            assert image.mode == 'RGB' and image.format == 'JPEG', row['id']
            assert image.size == (row['width'], row['height']), row['id']
            assert max(image.size) <= manifest['processing']['max_side'], row['id']
        label_rows = [line.split() for line in (destination/row['label']).read_text().splitlines() if line.strip()]
        assert len(label_rows) == row['count'], row['id']
        for fields in label_rows:
            assert len(fields) == 5 and fields[0] == '0', row['id']
            x, y, w, h = map(float, fields[1:])
            assert 0 < w <= 1 and 0 < h <= 1, row['id']
            assert min(x-w/2, y-h/2) >= -1e-5, row['id']
            assert max(x+w/2, y+h/2) <= 1+1e-5, row['id']
        for key in [('flight',row['flight']), ('farm_date',row['farm'],row['date'])]:
            assert key not in groups or groups[key] == row['split'], key
            groups[key] = row['split']
    for field in ('id','original_sha256','derived_sha256'):
        assert len({row[field] for row in rows}) == len(rows), field
    assert {r['farm'] for r in rows if r['split']=='test'} == {'Derval'}
    assert not any(r['farm']=='Derval' and r['split']!='test' for r in rows)
    for split in ('train','val','test'):
        selected = [r for r in rows if r['split']==split]
        actual_images = sorted((destination/'images'/split).glob('*.jpg'))
        actual_labels = sorted((destination/'labels'/split).glob('*.txt'))
        assert len(actual_images) == len(actual_labels) == len(selected), split
        counts[split] = {'images':len(selected), 'positive':sum(r['count']>0 for r in selected),
                         'boxes':sum(r['count'] for r in selected)}
    assert counts == manifest['counts'], (counts,manifest['counts'])
    packages = json.loads((ROOT/'assets'/'pacotes.json').read_text())
    assert all(p['size'] < 40_000_000 for p in packages['packages'])
    report = {'status':'passed','scope':'Data integrity only; no model evaluation.',
              'manifest_sha256':sha256(manifest_path.read_bytes()),
              'decoded_jpegs':len(rows), 'verified_label_files':len(rows),
              'verified_sample_files':2*len(rows), 'counts':counts,
              'packages':len(packages['packages']), 'package_total_bytes':packages['total_size'],
              'largest_package_bytes':max(p['size'] for p in packages['packages']),
              'duplicate_original_hashes':0,'duplicate_derived_hashes':0,
              'cross_split_flight_groups':0,'cross_split_farm_date_groups':0}
    (ROOT/'assets'/'validacao-dados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
