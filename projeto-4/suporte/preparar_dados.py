#!/usr/bin/env python3
"""Recria o subconjunto didático ICAERUS usando HTTP Range, sem ZIP de 16,6 GB.

Aluno: importar obter_dados() para extrair os pacotes que acompanham o projeto.
Reconstrução da fonte: python suporte/preparar_dados.py --destino dados/gado-icaerus-v1
Distribuição: acrescente --empacotar após concluir a reconstrução.
Requer Pillow. Até quatro leituras públicas simultâneas; sem conta ou API key.
O manifesto controla seleção, grupos, anotações e CRC de cada imagem original.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import subprocess
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]


def obter_dados(destino: str | Path | None = None) -> Path:
    """Extrai os pacotes didáticos locais verificados e retorna a raiz YOLO.

    Fluxo do aluno: nenhum download remoto nem credencial. Os pacotes são
    versionados com o projeto; esta função valida SHA-256 antes de extrair.
    """
    destination = Path(destino).resolve() if destino is not None else ROOT/'dados'/'gado'
    packages_path = ROOT/'assets'/'pacotes.json'
    if not packages_path.exists():
        raise FileNotFoundError('Pacotes ainda não disponíveis; use a distribuição completa do projeto.')
    packages = json.loads(packages_path.read_text())
    manifest_bytes = (ROOT/'assets'/'manifesto.json').read_bytes()
    if sha256(manifest_bytes) != packages['manifest_sha256']:
        raise RuntimeError('Manifesto diferente daquele usado para empacotar os dados.')
    manifest = json.loads(manifest_bytes)
    if packages['dataset'] != manifest['dataset']:
        raise RuntimeError('Versões dos dados e pacotes são diferentes.')
    destination.mkdir(parents=True,exist_ok=True)
    expected = {s[k] for s in manifest['samples'] for k in ('image', 'label')}
    # Ultralytics cria labels/train.cache e labels/val.cache após o treino.
    # Metadados locais não são imagens/rótulos extras nem alteram o manifesto.
    existing = {str(p.relative_to(destination)) for folder in ('images', 'labels')
                for p in (destination/folder).rglob('*') if p.is_file()
                and p.suffix != '.cache' and p.name != '.DS_Store'}
    extras = existing - expected
    if extras:
        raise RuntimeError('Destino contém imagens/rótulos de outra seleção; escolha uma pasta vazia. '
                           f'Exemplo: {sorted(extras)[0]}')
    for package in packages['packages']:
        archive_path = ROOT/'assets'/package['filename']
        if not archive_path.exists():
            raise FileNotFoundError(f'Pacote necessário ausente: {archive_path}')
        if archive_path.stat().st_size != package['size'] or sha256(archive_path.read_bytes()) != package['sha256']:
            raise RuntimeError(f'Pacote corrompido: {archive_path.name}; baixe novamente o repositório.')
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                target=(destination/info.filename).resolve()
                if destination not in target.parents:
                    raise RuntimeError(f'Caminho inválido dentro do ZIP: {info.filename}')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Links simbólicos não são aceitos no pacote.')
            archive.extractall(destination)
    for sample in manifest['samples']:
        for path_key,hash_key in [('image','derived_sha256'),('label','label_sha256')]:
            path=destination/sample[path_key]
            if not path.exists() or sha256(path.read_bytes()) != sample[hash_key]:
                raise RuntimeError(f'Arquivo ausente/corrompido após extração: {path}')
    (destination/'manifesto.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (destination/'data.yaml').write_text(f'path: {json.dumps(str(destination))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cow\n')
    return destination


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_member(source: dict, row: dict) -> bytes:
    """Lê um membro ZIP por faixa e verifica deflate, tamanho e CRC32."""
    start = row['zip_header_offset']
    # 4 KiB bastam para o header conhecido; o comprimento real é validado abaixo.
    end = min(source['archive_size'] - 1, start + row['compressed_size'] + 4095)
    expected = end - start + 1
    proc = subprocess.run([
        'curl', '-fsSL', '--max-time', '90', '--max-filesize', str(expected),
        '--range', f'{start}-{end}', '--write-out', '%{http_code}', source['url'],
    ], capture_output=True, check=True)
    data, status = proc.stdout[:-3], proc.stdout[-3:]
    if status != b'206' or len(data) != expected:
        raise RuntimeError(f'HTTP Range inválido: status={status!r}, tamanho={len(data)}')
    header = struct.unpack('<IHHHHHIIIHH', data[:30])
    signature, _, flags, method, *_ = header
    name_length, extra_length = header[-2:]
    if signature != 0x04034B50 or flags & 1:
        raise RuntimeError('Header ZIP inesperado ou criptografado')
    name_bytes = data[30:30 + name_length]
    name = name_bytes.decode('utf-8' if flags & 0x800 else 'cp437')
    if name != row['original_path']:
        raise RuntimeError(f'Membro ZIP diferente do manifesto: {name}')
    offset = 30 + name_length + extra_length
    payload = data[offset:offset + row['compressed_size']]
    if len(payload) != row['compressed_size']:
        raise RuntimeError('Membro ZIP truncado')
    original = zlib.decompress(payload, -15) if method == 8 else payload if method == 0 else None
    if original is None:
        raise RuntimeError(f'Compressão ZIP não suportada: {method}')
    if len(original) != row['original_size']:
        raise RuntimeError('Tamanho original diferente do inventário')
    if f'{zlib.crc32(original):08x}' != row['original_crc32']:
        raise RuntimeError('CRC32 original diferente do inventário')
    return original


def write_metadata(destination: Path, source_manifest: dict, records: list) -> None:
    manifest = {k:v for k,v in source_manifest.items() if k != 'images'}
    manifest['build'] = {'pillow_version':pillow_version,
                         'source_manifest_sha256':sha256(json.dumps(source_manifest,sort_keys=True,ensure_ascii=False).encode()),
                         'complete':len(records)==len(source_manifest['images'])}
    manifest['images'] = records
    target = destination/'manifesto.json'
    temp = destination/'manifesto.json.tmp'
    temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    temp.replace(target)


def empacotar(destination: Path, assets: Path | None = None) -> dict:
    """Congela uma reconstrução completa em partes verificáveis menores que 40 MB.

    Usa ZIP com nomes/ordem/data fixos. Os JPEGs distribuídos são os bytes da
    reconstrução validada; o aluno não depende da versão local do codificador JPEG.
    """
    assets = assets or ROOT/'assets'
    built = json.loads((destination/'manifesto.json').read_text())
    if not built.get('build', {}).get('complete'):
        raise RuntimeError('Reconstrução incompleta; não publicar um pacote parcial.')
    records = built.get('images', built.get('samples', []))
    if len(records) != 300:
        raise RuntimeError('Este pacote exige exatamente as 300 imagens da seleção v1.')
    samples = []
    for record in records:
        sample = dict(record)
        sample.update(id=Path(record.get('image_path', record.get('image'))).stem,
                      image=record.get('image_path', record.get('image')),
                      label=record.get('label_path', record.get('label')),
                      count=record.get('n_boxes', record.get('count')))
        for key, digest in [('image', 'derived_sha256'), ('label', 'label_sha256')]:
            if sha256((destination/sample[key]).read_bytes()) != sample[digest]:
                raise RuntimeError(f'Hash divergente durante empacotamento: {sample[key]}')
        samples.append(sample)
    if len({s['id'] for s in samples}) != len(samples):
        raise RuntimeError('IDs duplicados na seleção.')
    if len({s['original_sha256'] for s in samples}) != len(samples):
        raise RuntimeError('Imagens originais idênticas na seleção; revisar antes de publicar.')
    if len({s['derived_sha256'] for s in samples}) != len(samples):
        raise RuntimeError('Imagens derivadas idênticas na seleção; revisar antes de publicar.')
    manifest = {k:v for k,v in built.items() if k not in ('images', 'samples')}
    manifest['source_manifest_file_sha256'] = sha256((ROOT/'assets'/'manifesto-fonte-icaerus-v1.json').read_bytes())
    manifest['source_audit_file_sha256'] = sha256((ROOT/'assets'/'auditoria-fonte.json').read_bytes())
    manifest['samples'] = samples
    manifest['counts'] = {split: {'images': sum(s['split']==split for s in samples),
                                 'positive': sum(s['split']==split and s['count']>0 for s in samples),
                                 'boxes': sum(s['count'] for s in samples if s['split']==split)}
                          for split in ('train', 'val', 'test')}
    groups = {}
    for sample in samples:
        for group in (('farm_flight', sample['farm'], sample['flight']),
                      ('flight', sample['flight']),
                      ('farm_date', sample['farm'], sample['date'])):
            if group in groups and groups[group] != sample['split']:
                raise RuntimeError(f'Grupo de captura cruza partições: {group}')
            groups[group] = sample['split']
    manifest['integrity_audit'] = {
        'samples_verified':len(samples), 'original_exact_duplicates':0,
        'derived_exact_duplicates':0, 'cross_split_flight_groups':0,
        'cross_split_farm_date_groups':0,
        'test_farms':sorted({s['farm'] for s in samples if s['split']=='test'}),
        'individual_animal_identity_available':False,
        'method':'Exact SHA-256, labels, farm/flight and farm/date membership; no visual assessment of test data used to choose parameters.'}
    assets.mkdir(parents=True, exist_ok=True)
    files = ['ATRIBUICAO.md']
    files += [s[k] for s in samples for k in ('image','label')]
    chunks, current, size = [], [], 0
    for name in files:
        # Margem para headers ZIP e expansão mínima eventual do deflate.
        contribution = (destination/name).stat().st_size + 512 + len(name.encode('utf-8'))*2
        if size + contribution > 38_000_000 and current:
            chunks.append(current)
            current, size = [], 0
        current.append(name)
        size += contribution
    if current:
        chunks.append(current)
    packages = []
    for i, names in enumerate(chunks, 1):
        path = assets/f'gado-icaerus-v1-parte-{i:02d}.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (destination/name).read_bytes(), compresslevel=6)
        size = path.stat().st_size
        if size >= 40_000_000:
            raise RuntimeError(f'Pacote maior que o limite de distribuição: {path}')
        packages.append({'filename':path.name, 'size':size,
                         'sha256':sha256(path.read_bytes()), 'files':len(names)})
    manifest_path = assets/'manifesto.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    package_index = {'schema_version':1, 'dataset':manifest['dataset'],
                     'manifest_sha256':sha256(manifest_path.read_bytes()),
                     'total_size':sum(p['size'] for p in packages), 'packages':packages}
    (assets/'pacotes.json').write_text(json.dumps(package_index,ensure_ascii=False,indent=2)+'\n')
    return package_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifesto',type=Path,default=ROOT/'assets'/'manifesto-fonte-icaerus-v1.json')
    parser.add_argument('--destino',type=Path,default=ROOT/'dados'/'gado-icaerus-v1')
    parser.add_argument('--limite',type=int,default=None,help='Smoke test: baixar somente as N primeiras imagens; não usar como treino final.')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=4,help='Até quatro leituras públicas do Zenodo; não é transporte do Campus.')
    parser.add_argument('--empacotar',action='store_true',help='Somente empacotar a reconstrução já concluída em --destino.')
    args = parser.parse_args()
    if args.empacotar:
        result = empacotar(args.destino.resolve())
        print(f"Pacotes prontos: {len(result['packages'])}; {result['total_size']/1e6:.1f} MB")
        return
    manifest = json.loads(args.manifesto.read_text())
    if args.limite is not None and not 1 <= args.limite <= len(manifest['images']):
        parser.error('--limite deve estar entre 1 e o tamanho do manifesto.')
    destination = args.destino.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    previous = {}
    if (destination/'manifesto.json').exists():
        old=json.loads((destination/'manifesto.json').read_text())
        if old['dataset'] != manifest['dataset'] or old['processing'] != manifest['processing']:
            raise RuntimeError('Destino contém outra versão; use uma pasta vazia.')
        if old.get('build',{}).get('pillow_version') != pillow_version:
            raise RuntimeError('Destino usa outra versão de Pillow; preserve a versão ou use uma pasta vazia.')
        previous_source_hash = old.get('build',{}).get('source_manifest_sha256')
        if previous_source_hash and previous_source_hash != sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode()):
            raise RuntimeError('A seleção de origem mudou; use outra versão e pasta de destino.')
        previous={r['original_path']:r for r in old['images']}
    rows=manifest['images'][:args.limite] if args.limite else manifest['images']
    records=[]
    start_time=time.monotonic()
    def prepare_one(row):
        image_path=destination/row['image_path']
        label_path=destination/row['label_path']
        old=previous.get(row['original_path'])
        if old and image_path.exists() and label_path.exists():
            if sha256(image_path.read_bytes())==old['derived_sha256'] and sha256(label_path.read_bytes())==old['label_sha256']==sha256(row['label_yolo'].encode()):
                return old
        original=read_member(manifest['source'],row)
        with Image.open(io.BytesIO(original)) as image:
            original_width,original_height=image.size
            image=image.convert('RGB')
            image.thumbnail((manifest['processing']['max_side'],)*2,Image.Resampling.LANCZOS)
            buffer=io.BytesIO()
            image.save(buffer,format='JPEG',quality=manifest['processing']['jpeg_quality'],optimize=True)
            derived=buffer.getvalue()
            derived_width,derived_height=image.size
        image_path.parent.mkdir(parents=True,exist_ok=True)
        label_path.parent.mkdir(parents=True,exist_ok=True)
        image_path.write_bytes(derived)
        label_data=row['label_yolo'].encode()
        # Um TXT vazio existe na fonte: isto é diferente de um TXT ausente.
        label_path.write_bytes(label_data)
        record={k:v for k,v in row.items() if k != 'label_yolo'}
        record.update(original_sha256=sha256(original),derived_sha256=sha256(derived),label_sha256=sha256(label_data),
                      original_width=original_width,original_height=original_height,
                      width=derived_width,height=derived_height,derived_size=len(derived))
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i,record in enumerate(executor.map(prepare_one,rows),1):
            records.append(record)
            write_metadata(destination,manifest,records)
            print(f'{i}/{len(rows)} {record["split"]} {record["farm"]}: {record["n_boxes"]} caixas; {record["derived_size"]//1024} KiB; {time.monotonic()-start_time:.0f}s',flush=True)
    write_metadata(destination,manifest,records)
    (destination/'data.yaml').write_text(f'path: {json.dumps(str(destination))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cow\n')
    (destination/'ATRIBUICAO.md').write_text(
        '# Dados de origem\n\nLouise Helary e Adrien Lebreton. *Drone images and their annotations of grazing cows*, v2 (2024). '
        'Institut de l’Elevage / ICAERUS. DOI: https://doi.org/10.5281/zenodo.11048412.\n\n'
        'Licença CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.\n\n'
        'Adaptação didática Sigmoidal: seleção determinística, divisão por fazenda/data/voo, '
        'redimensionamento proporcional para lado máximo de 2048 pixels e recompressão JPEG qualidade 85. '
        'Coordenadas YOLO normalizadas e classe cow preservadas. '
        'Não há endosso dos autores originais. Os dados derivados continuam sob CC BY 4.0.\n')
    print(f'Dataset preparado em {destination}; completo={len(records)==len(manifest["images"])}')


if __name__=='__main__':
    main()
