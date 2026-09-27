#!/usr/bin/env python3
"""Reconstrói e empacota somente Other_farms, sem modificar o conjunto principal.

Antes: python scripts/remote_inventory.py && python scripts/auditar_fonte.py
Uso: python scripts/preparar_reserva.py [--workers 4] [--empacotar]
Não visualizar nem avaliar esta reserva antes de congelar a receita do modelo.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, __version__ as pillow_version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from suporte.preparar_dados import read_member,sha256


def source_manifest():
    cached=ROOT/'dados'/'.auditoria-fonte'
    rows=json.loads((cached/'icaerus_image_metadata.json').read_text())
    annotations=json.loads((cached/'icaerus_annotations.json').read_text())
    base=json.loads((ROOT/'assets'/'manifesto-fonte-icaerus-v1.json').read_text())
    selected=[]
    for row in sorted((r for r in rows if r['farm']=='Other_farms'),key=lambda r:r['path']):
        stem='other_farms__'+Path(row['path']).stem
        selected.append({'id':stem,'original_path':row['path'],'original_label_path':row['label_path'],
                         'farm':row['farm'],'flight':row['flight'],'date':row['date'],'split':'test',
                         'image':f'images/test/{stem}.jpg','label':f'labels/test/{stem}.txt',
                         'count':row['n_boxes'],'label_yolo':annotations[row['label_path']],
                         'zip_header_offset':row['header_offset'],'compressed_size':row['compressed_size'],
                         'original_size':row['size'],'original_crc32':row['crc32']})
    if len(selected)!=62 or sum(r['count'] for r in selected)!=1355 or any(r['count']==0 for r in selected):
        raise RuntimeError('A fonte não corresponde à reserva esperada; interromper e revisar.')
    result={'schema_version':1,'dataset':'reserva-icaerus-v1','source':base['source'],
            'processing':base['processing'],'role':'external_holdout_after_recipe_freeze',
            'split_policy':'All 62 Other_farms images are held out as test; no train or validation split.',
            'scope_notes':['Other_farms is an aggregated source folder, not a verified single farm.',
                           'Geographical independence from the principal dataset is not established.',
                           'All 62 images are positive; no empty-pasture negative evaluation is possible.',
                           'No visual inspection or model inference may precede recipe freezing.'],
            'counts':{'test':{'images':62,'positive':62,'boxes':1355}},'images':selected}
    (ROOT/'assets'/'reserva-fonte.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def package(destination,manifest):
    records=manifest['samples']
    if not manifest['build']['complete'] or len(records)!=62:
        raise RuntimeError('Reserva incompleta; não empacotar.')
    base=json.loads((ROOT/'assets'/'manifesto.json').read_text())
    for field in ('original_sha256','derived_sha256'):
        hashes={r[field] for r in records}
        if len(hashes)!=62 or hashes & {r[field] for r in base['samples']}:
            raise RuntimeError(f'Imagens repetidas ou compartilhadas com base: {field}')
    for field in ('flight',):
        if {r[field] for r in records} & {r[field] for r in base['samples']}:
            raise RuntimeError('Voo da reserva já presente no conjunto principal.')
    manifest['integrity_audit']={'unique_original_hashes':62,'unique_derived_hashes':62,
                                 'shared_image_hashes_with_base':0,'shared_flights_with_base':0,
                                 'visual_inspection_performed':False,'model_inference_performed':False}
    manifest['source_manifest_sha256']=sha256((ROOT/'assets'/'reserva-fonte.json').read_bytes())
    manifest_path=ROOT/'assets'/'reserva-manifesto.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (destination/'ATRIBUICAO.md').write_text(
        '# Reserva de dados ICAERUS\n\nLouise Helary e Adrien Lebreton. '
        '*Drone images and their annotations of grazing cows*, v2 (2024). '
        'Institut de l’Elevage / ICAERUS. DOI: https://doi.org/10.5281/zenodo.11048412.\n\n'
        'Licença CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.\n\n'
        'Adaptação Sigmoidal: todas as 62 imagens da pasta Other_farms, redução proporcional '
        'para lado máximo 2048 pixels, JPEG qualidade 85 e rótulos YOLO preservados. '
        'Other_farms agrupa origens não individualizadas. Todas as imagens são positivas. '
        'Não há endosso dos autores originais. Os dados derivados continuam sob CC BY 4.0.\n')
    files=['ATRIBUICAO.md']+[r[k] for r in records for k in ('image','label')]
    chunks=[];current=[];size=0
    for name in files:
        contribution=(destination/name).stat().st_size+1024
        if current and size+contribution>38_000_000:
            chunks.append(current);current=[];size=0
        current.append(name);size+=contribution
    if current:chunks.append(current)
    packages=[]
    for i,names in enumerate(chunks,1):
        path=ROOT/'assets'/f'reserva-icaerus-v1-parte{i:02d}.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in names:
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o100644<<16
                archive.writestr(info,(destination/name).read_bytes(),compresslevel=6)
        if path.stat().st_size>=40_000_000:raise RuntimeError('Parte da reserva excede40MB.')
        packages.append({'filename':path.name,'size':path.stat().st_size,'sha256':sha256(path.read_bytes()),'files':len(names)})
    index={'schema_version':1,'dataset':manifest['dataset'],'manifest_sha256':sha256(manifest_path.read_bytes()),
           'total_size':sum(p['size'] for p in packages),'packages':packages}
    (ROOT/'assets'/'reserva-pacotes.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(f"Reserva empacotada: {len(packages)} partes, {index['total_size']/1e6:.1f} MB",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--empacotar',action='store_true',help='Apenas empacotar reconstrução completa.')
    args=parser.parse_args()
    destination=ROOT/'dados'/'reserva-fonte';destination.mkdir(parents=True,exist_ok=True)
    path=destination/'manifesto.json'
    if args.empacotar:
        package(destination,json.loads(path.read_text()));return
    if pillow_version!='11.3.0':raise RuntimeError('Reconstrução v1 requer Pillow11.3.0.')
    source=source_manifest();previous={}
    if path.exists():
        old=json.loads(path.read_text())
        if old['dataset']!=source['dataset'] or old['processing']!=source['processing']:
            raise RuntimeError('Destino já contém outra versão da reserva.')
        previous={r['original_path']:r for r in old['samples']}
    records=[];started=time.monotonic()
    def prepare(row):
        image_path=destination/row['image'];label_path=destination/row['label'];old=previous.get(row['original_path'])
        if old and image_path.exists() and label_path.exists():
            if sha256(image_path.read_bytes())==old['derived_sha256'] and sha256(label_path.read_bytes())==old['label_sha256']==sha256(row['label_yolo'].encode()):return old
        original=read_member(source['source'],row)
        with Image.open(io.BytesIO(original)) as image:
            ow,oh=image.size;image=image.convert('RGB');image.thumbnail((2048,2048),Image.Resampling.LANCZOS)
            data=io.BytesIO();image.save(data,format='JPEG',quality=85,optimize=True);derived=data.getvalue();w,h=image.size
        image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
        image_path.write_bytes(derived);labels=row['label_yolo'].encode();label_path.write_bytes(labels)
        record={k:v for k,v in row.items() if k!='label_yolo'}
        record.update(original_sha256=sha256(original),derived_sha256=sha256(derived),label_sha256=sha256(labels),
                      original_width=ow,original_height=oh,width=w,height=h,derived_size=len(derived))
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(prepare,row) for row in source['images']]
        for completed,future in enumerate(as_completed(futures),1):
            records.append(future.result())
            manifest={k:v for k,v in source.items() if k!='images'}
            manifest['samples']=sorted(records,key=lambda r:r['original_path'])
            manifest['build']={'pillow_version':pillow_version,'complete':completed==62}
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
            if completed%5==0 or completed==62:print(f'Reserva {completed}/62; {time.monotonic()-started:.0f}s',flush=True)
    package(destination,manifest)


if __name__=='__main__':main()
