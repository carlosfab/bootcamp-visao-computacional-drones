"""Suporte aos Notebooks 02 e 03: download e carregamento verificado, sem inferência.

O SHA-256 foi medido localmente no arquivo público; não foi publicado pelos autores.
O código oficial precisa estar no commit indicado, com seus arquivos intactos.
"""
from pathlib import Path
import ast
import hashlib
import pickle
import pickletools
import subprocess
import sys
import types
import zipfile

try:
    from .downloads import recusar_temporarios_gdown, transferir_gdown
except ImportError:
    from downloads import recusar_temporarios_gdown, transferir_gdown

URL_OFICIAL = 'https://drive.google.com/file/d/1Savf3oeiWek4eeXrvYLuaoBMoZW3nag8/view'
SHA256_VTUAV = '0c6d399dfe902d7e3d57b6a3d62fc0ff9c286e656568a5c2d2cd4fd21923513a'
COMMIT_OFICIAL = 'a79210c283597294ded255028bb8fe16e1c241cd'
CONFIG_OFICIAL = 'qfdet_configs/qfdet_r50_fpn_1x_vtuav.py'
GLOBAIS = {('collections', 'OrderedDict'), ('torch', 'FloatStorage'),
           ('torch', 'LongStorage'), ('torch._utils', '_rebuild_tensor_v2')}
INATIVOS = {f'fuse.{ramo}.{parametro}' for ramo in (
    'fusion_cat2.conv1x1', 'fusion_gated.t_conv1x1', 'fusion_gated.v_conv1x1')
    for parametro in ('weight', 'bias')}


def _exigir(condicao, mensagem):
    if not condicao:
        raise ValueError(mensagem)


def _sha256(caminho):
    digest = hashlib.sha256()
    with Path(caminho).open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(8 * 1024 * 1024), b''):
            digest.update(bloco)
    return digest.hexdigest()


def baixar_peso(destino: Path, url: str, sha256: str) -> Path:
    """Reutiliza cache íntegro ou baixa uma vez; falhas não são repetidas."""
    destino = Path(destino)
    sha256 = sha256.lower()
    _exigir(len(sha256) == 64 and set(sha256) <= set('0123456789abcdef'), 'SHA-256 inválido.')
    if destino.exists():
        _exigir(_sha256(destino) == sha256, 'O peso em cache não corresponde ao SHA-256 esperado.')
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    recusar_temporarios_gdown(destino)
    parcial = destino.with_name(destino.name + '.part')
    _exigir(not parcial.exists(), f'Existe um download parcial anterior: {parcial}')
    transferir_gdown(parcial, url=url, fuzzy=True)
    _exigir(parcial.is_file(), 'O download não foi concluído.')
    _exigir(_sha256(parcial) == sha256, 'SHA-256 incorreto; o arquivo .part não será utilizado.')
    parcial.replace(destino)
    return destino


class _UnpicklerRestrito(pickle.Unpickler):
    def find_class(self, modulo, nome):
        if (modulo, nome) not in GLOBAIS:
            raise pickle.UnpicklingError(f'Global bloqueado: {modulo}.{nome}')
        return super().find_class(modulo, nome)


def _ler_pickle(arquivo, **kwargs):
    return _UnpicklerRestrito(arquivo, **kwargs).load()


def ler_checkpoint_restrito(peso):
    """Lê tensores na CPU com a mesma lista restrita nos Notebooks 02 e 03.

    O chamador deve conferir o SHA-256 registrado no projeto antes desta leitura.
    """
    with zipfile.ZipFile(peso) as arquivo:
        _exigir(arquivo.testzip() is None, 'Falha de CRC no checkpoint.')
        nomes = [nome for nome in arquivo.namelist() if nome.endswith('.pkl')]
        _exigir(nomes == ['archive/data.pkl'], 'Estrutura pickle inesperada.')
        ops = list(pickletools.genops(arquivo.read(nomes[0])))
        _exigir(all(tuple(arg.split(' ', 1)) in GLOBAIS for op, arg, _ in ops if op.name == 'GLOBAL')
                and not any(op.name in {'STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4'} for op, _, _ in ops),
                'O pickle contém instruções não permitidas.')
    import torch
    modulo_pickle = types.SimpleNamespace(__name__='pickle_restrito',
                                         Unpickler=_UnpicklerRestrito, load=_ler_pickle)
    return torch.load(peso, map_location='cpu', pickle_module=modulo_pickle)


def _config_literal(texto):
    """Interpreta somente literais, referências anteriores e dict(...)."""
    valores = {}
    def avaliar(no):
        if isinstance(no, ast.Constant):
            return no.value
        if isinstance(no, (ast.List, ast.Tuple)):
            itens = [avaliar(item) for item in no.elts]
            return tuple(itens) if isinstance(no, ast.Tuple) else itens
        if isinstance(no, ast.Dict):
            return {avaliar(k): avaliar(v) for k, v in zip(no.keys, no.values)}
        if isinstance(no, ast.Name) and no.id in valores:
            return valores[no.id]
        if isinstance(no, ast.UnaryOp) and isinstance(no.op, ast.USub):
            return -avaliar(no.operand)
        if (isinstance(no, ast.Call) and isinstance(no.func, ast.Name)
                and no.func.id == 'dict' and not no.args
                and all(k.arg is not None for k in no.keywords)):
            return {k.arg: avaliar(k.value) for k in no.keywords}
        raise ValueError(f'Configuração não literal: {ast.dump(no)}')
    for no in ast.parse(texto).body:
        _exigir(isinstance(no, ast.Assign) and len(no.targets) == 1
                and isinstance(no.targets[0], ast.Name), 'Instrução de configuração não permitida.')
        valores[no.targets[0].id] = avaliar(no.value)
    return valores


def carregar_modelo(fonte: Path, peso: Path):
    """Retorna (modelo CPU/eval, configuração, relatório); não executa inferência."""
    fonte, peso = Path(fonte).resolve(), Path(peso).resolve()
    revisao = subprocess.check_output(['git', '-C', str(fonte), 'rev-parse', 'HEAD'], text=True).strip()
    _exigir(revisao == COMMIT_OFICIAL, 'O checkout QFDet não está no commit validado.')
    subprocess.run(['git', '-C', str(fonte), 'diff', '--exit-code', '--quiet', 'HEAD', '--',
                    'mmdet', 'qfdet_configs', 'configs/_base_'], check=True)
    _exigir(peso.stat().st_size == 485107939 and _sha256(peso) == SHA256_VTUAV,
            'O checkpoint não corresponde ao arquivo VTUAV-det validado.')
    sys.path.insert(0, str(fonte))
    import torch
    import mmdet
    from mmcv import Config
    from mmdet.models import build_detector
    _exigir(Path(mmdet.__file__).resolve().parent == fonte / 'mmdet',
            'Outro MMDetection está importado; reinicie o kernel com o checkout correto.')
    checkpoint = ler_checkpoint_restrito(peso)
    meta = checkpoint['meta']
    _exigir(tuple(meta['CLASSES']) == ('person',), 'Classes do checkpoint inesperadas.')
    embutida = _config_literal(meta['config'])
    cfg = Config.fromfile(str(fonte / CONFIG_OFICIAL))
    _exigir(embutida['dataset_type'] == cfg.dataset_type == 'VTUAVdet', 'Dataset incompatível.')
    _exigir({k: cfg.img_norm_cfg[k] for k in embutida['img_norm_cfg']} == embutida['img_norm_cfg'],
            'Normalização incompatível.')
    cfg.img_norm_cfg = embutida['img_norm_cfg']  # Remove apenas globais COCO herdadas e não utilizadas.
    _exigir(cfg.test_pipeline == embutida['test_pipeline'], 'Pipeline incompatível.')
    for campo in ('neck', 'backbone', 'bbox_head', 'test_cfg', 'poolupsample'):
        _exigir(cfg.model[campo] == embutida['model'][campo], f'Configuração incompatível: {campo}')
    antigo, atual = dict(embutida['model']['bbox_prehead']), dict(cfg.model.bbox_prehead)
    _exigir(antigo.pop('type') == 'ATSSQPreLCHead' and atual.pop('type') == 'QFDetPreHead'
            and antigo == atual, 'Pré-classificador incompatível.')
    _exigir(embutida['model']['type'] == 'ATSSQLCF' and cfg.model.type == 'QFDet'
            and cfg.model.base_fusion == 'cat' and cfg.model.quality_attention is True
            and cfg.model.poolupsample == 1 and cfg.model.reweight is True
            and cfg.model.test_cfg.nms.iou_threshold == 0.5, 'Arquitetura ou parâmetros inesperados.')
    estado = checkpoint['state_dict']
    for cabeca in ('bbox_head', 'bbox_prehead'):
        _exigir(cfg.model[cabeca].num_classes == 3
                and tuple(estado[f'{cabeca}.atss_cls.weight'].shape) == (3, 256, 3, 3),
                'Os três canais físicos do checkpoint devem ser preservados.')
    cfg.model.train_cfg = None
    cfg.model.backbone.init_cfg = None  # Impede download de pesos adicionais do backbone.
    modelo = build_detector(cfg.model).cpu().eval()
    esperado = modelo.state_dict()
    _exigir(not (set(esperado) - set(estado)) and set(estado) - set(esperado) == INATIVOS,
            'Diferenças inesperadas entre o checkpoint e o modelo.')
    ativos = {k: v for k, v in estado.items() if k not in INATIVOS}
    _exigir(all(v.shape == esperado[k].shape for k, v in ativos.items()), 'Shapes incompatíveis.')
    modelo.load_state_dict(ativos, strict=True)
    modelo.CLASSES = ('person',)
    relatorio = dict(official_url=URL_OFICIAL, sha256=SHA256_VTUAV, code_revision=revisao,
                    strict_load=True, active_state_entries=len(ativos), discarded_inactive_keys=sorted(INATIVOS),
                    physical_head_outputs=3, dataset_classes=['person'], labeled_output_indices=[0],
                    unlabeled_output_indices=[1, 2], image_order=['visible', 'thermal'],
                    annotation_reference='thermal/IR', image_normalization=dict(cfg.img_norm_cfg),
                    test_cfg=dict(cfg.model.test_cfg), reweight=cfg.model.reweight,
                    restricted_unpickler=True, embedded_config_executed=False, inference_executed=False)
    return modelo, cfg, relatorio
