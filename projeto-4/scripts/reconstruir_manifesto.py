#!/usr/bin/env python3
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1]
HERE = TARGET / "dados" / ".auditoria-fonte"
rows = json.loads((HERE / 'icaerus_image_metadata.json').read_text())
annotations = json.loads((HERE / 'icaerus_annotations.json').read_text())
excluded = 'DJI_20230726084333_0004_V'

def rank(row):
    return hashlib.sha256(('sigmoidal-gado-v1|' + row['path']).encode()).hexdigest()

def evenly(candidates, n):
    groups = defaultdict(list)
    for row in candidates:
        groups[(row['farm'], row['flight'])].append(row)
    for group in groups.values():
        group.sort(key=rank)
    chosen = []
    while len(chosen) < n:
        advanced = False
        for key in sorted(groups):
            if groups[key]:
                chosen.append(groups[key].pop(0))
                advanced = True
                if len(chosen) == n:
                    break
        if not advanced:
            raise ValueError('Insufficient candidates')
    return chosen

splits = defaultdict(list)
for row in rows:
    if Path(row['path']).stem == excluded:
        continue
    if row['farm'] == 'Derval':
        split = 'test'
    elif row['farm'] in ('Mauron', 'Jalogny'):
        split = 'train' if row['date'] < '20230901' else 'val'
    else:
        continue
    splits[split].append(row)

selected = []
for split, positive, empty in [('train',120,60),('val',18,42),('test',20,40)]:
    candidates = splits[split]
    chosen = evenly([r for r in candidates if r['n_boxes'] > 0], positive)
    chosen += evenly([r for r in candidates if r['n_boxes'] == 0], empty)
    for row in sorted(chosen, key=lambda r: r['path']):
        stem = row['farm'].lower() + '__' + Path(row['path']).stem
        selected.append({
            'original_path':row['path'], 'original_label_path':row['label_path'],
            'farm':row['farm'], 'flight':row['flight'], 'date':row['date'], 'split':split,
            'image_path':f'images/{split}/{stem}.jpg', 'label_path':f'labels/{split}/{stem}.txt',
            'n_boxes':row['n_boxes'], 'label_yolo':annotations[row['label_path']],
            'zip_header_offset':row['header_offset'], 'compressed_size':row['compressed_size'],
            'original_size':row['size'], 'original_crc32':row['crc32'],
        })

manifest = {
    'schema_version':1, 'dataset':'gado-icaerus-v1', 'selection_seed':'sigmoidal-gado-v1',
    'source':{'title':'Drone images and their annotations of grazing cows',
              'authors':['Louise Helary','Adrien Lebreton'], 'institution':"Institut de l’Elevage / ICAERUS",
              'doi':'10.5281/zenodo.11048412', 'version':'v2', 'license':'CC-BY-4.0',
              'url':'https://zenodo.org/api/records/11048412/files/Cattle_drone_images_042024.zip/content',
              'archive_size':16586848207, 'archive_md5':'c18911fb11cb58741b9cfa16516ca8cd'},
    'processing':{'max_side':2048,'jpeg_quality':85,'pillow_resample':'LANCZOS',
                  'exif_orientation':'Do not auto-rotate; preserve original pixel coordinates.',
                  'classes':{'0':'cow'},'labels':'Original normalized YOLO coordinates and class 0 cow preserved.'},
    'split_policy':'train: Mauron/Jalogny before 2023-09-01; val: same farms after that date; test: Derval only. No farm-date or flight appears in multiple splits. Animals are not individually identified.',
    'sampling_policy':'Within each split and positive/negative stratum, round-robin across sorted farm/flight groups; rank images by SHA256(selection_seed|original_path). Training positives deliberately enriched; held-out splits approximately preserve source prevalence.',
    'audit':{'source_images':1385,'source_flights':25,'source_yolo_boxes':4747,
             'source_advertised_boxes':4941,'missing_yolo_labels':0,'ambiguous_matches':0,
             'excluded_image_stems':[excluded],
             'exclusion_reason':'VOC contains 15 objects (including a non-cow label) while YOLO contains 14; exclude rather than repair without source review.'},
    'images':selected,
}
(TARGET/'assets').mkdir(parents=True,exist_ok=True)
path=TARGET/'assets'/'manifesto-fonte-icaerus-v1.json'
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(path)
print(json.dumps({s:{'images':sum(r['split']==s for r in selected), 'positive':sum(r['split']==s and r['n_boxes']>0 for r in selected), 'boxes':sum(r['n_boxes'] for r in selected if r['split']==s), 'source_gb':sum(r['compressed_size'] for r in selected if r['split']==s)/1e9} for s in splits},indent=2))
