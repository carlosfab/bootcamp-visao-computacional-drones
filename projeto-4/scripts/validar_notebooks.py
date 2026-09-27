"""Executa os notebooks reais, registra ambiente e empacota a conferência.

Executar na raiz de projeto-4. Uso: python scripts/validar_notebooks.py.
O notebook de treino exige pasta nova; para repetir, escolha outro nome nele.
Os notebooks executados, relatórios e pacote de conferência são atualizados.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')
RESULTS = ROOT / 'resultados' / 'validacao'
RESULTS.mkdir(parents=True, exist_ok=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(path):
    start = time.monotonic()
    log = RESULTS / (path.stem + '.log')
    with log.open('w') as output:
        completed = subprocess.run([
            sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
            '--inplace', '--ExecutePreprocessor.timeout=1800', str(path),
        ], stdout=output, stderr=subprocess.STDOUT)
    import nbformat
    notebook = nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    errors = [out for cell in code for out in cell.outputs if out.output_type == 'error']
    report = {'notebook': path.name, 'seconds': time.monotonic() - start,
              'code_cells': len(code), 'executed_cells': sum(c.execution_count is not None for c in code),
              'errors': len(errors), 'returncode': completed.returncode,
              'sha256_executed': digest(path)}
    (RESULTS / (path.stem + '.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if completed.returncode or errors or report['executed_cells'] != len(code):
        print(log.read_text()[-8000:], flush=True)
        raise RuntimeError(f'Notebook incompleto: {path.name}')
    for index, cell in enumerate(code):
        for number, out in enumerate(cell.outputs):
            png = out.get('data', {}).get('image/png')
            if png:
                import base64
                (RESULTS / f'{path.stem}_c{index:02d}_{number}.png').write_bytes(base64.b64decode(png))
    return report


def pack_weights():
    # O arquivo de conclusão vincula o treino do notebook 02 aos bytes exatos.
    # Mtime não identifica uma execução: cópias e treinos antigos o alteram.
    run = ROOT / 'resultados' / 'fine_tuning'
    conclusion = json.loads((run / 'treinamento_concluido.json').read_text())
    configuration = json.loads((run / 'configuracao_treino.json').read_text())
    best = Path(conclusion['weights'])
    best = (ROOT / best).resolve() if not best.is_absolute() else best.resolve()
    if not best.is_relative_to((ROOT / 'resultados' / 'treino').resolve()):
        raise RuntimeError('Checkpoint fora do diretório de treino deste projeto.')
    if best.name != 'best.pt' or not best.is_file():
        raise RuntimeError('A conclusão do notebook 02 não aponta para best.pt existente.')
    if digest(best) != conclusion['weights_sha256']:
        raise RuntimeError('O checkpoint mudou após a conclusão do notebook 02.')
    manifest_hash = digest(ROOT / 'assets' / 'manifesto.json')
    if manifest_hash != conclusion['manifest_sha256']:
        raise RuntimeError('O treino usou um manifesto diferente da distribuição.')
    if conclusion['epochs_completed'] != configuration['epochs']:
        raise RuntimeError('O treino não concluiu todas as épocas configuradas.')
    archive = ROOT / 'assets' / 'pesos-gado.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(best, 'pesos/gado.pt')
    metadata = {'archive_sha256': digest(archive), 'weights_sha256': digest(best),
                'architecture': 'YOLO11n', 'dataset': 'gado-icaerus-v1',
                'source': 'Treinamento do notebook 02; checkpoint selecionado na validação.',
                'base_weights': 'Ultralytics YOLO11n COCO', 'license': 'AGPL-3.0',
                'dataset_manifest_sha256': manifest_hash,
                'training_completion_sha256': digest(run / 'treinamento_concluido.json'),
                'epochs_completed': conclusion['epochs_completed'],
                'initial_weights_sha256': conclusion['initial_weights_sha256']}
    (ROOT / 'assets' / 'pesos-gado.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main():
    import torch
    packages = ['torch', 'torchvision', 'ultralytics', 'numpy', 'pandas', 'matplotlib',
                'Pillow', 'opencv-python', 'nbconvert', 'ipykernel']
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'cuda': torch.version.cuda,
                   'packages': {name: importlib.metadata.version(name) for name in packages}}
    (RESULTS / 'ambiente.json').write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment), flush=True)
    reports = []
    for name in ['00_dados_e_problema.ipynb', '01_baseline_e_contagem.ipynb', '02_fine_tuning.ipynb']:
        reports.append(execute(ROOT / name))
    pack_weights()
    reports.append(execute(ROOT / '03_objetos_pequenos.ipynb'))
    (RESULTS / 'execucoes.json').write_text(json.dumps(reports, indent=2))
    bundle = ROOT / 'validacao-gado.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in ROOT.glob('*.ipynb'):
            z.write(path, path.relative_to(ROOT))
        for path in (ROOT / 'resultados').rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.json', '.csv', '.png', '.log', '.yaml'}:
                z.write(path, path.relative_to(ROOT))
        for name in ['pesos-gado.zip', 'pesos-gado.json']:
            z.write(ROOT / 'assets' / name, 'assets/' + name)
    print('ARQUIVO_FINAL', bundle.name, bundle.stat().st_size, digest(bundle), flush=True)


if __name__ == '__main__':
    main()
