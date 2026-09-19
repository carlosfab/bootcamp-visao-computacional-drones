"""Testa a lógica real das células sem GPU, rede ou execução de treinamento."""
import ast
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJETO = Path(__file__).resolve().parents[1]


def notebook(nome):
    return json.loads((PROJETO / nome).read_text(encoding="utf-8"))


PREPARACAO = notebook("01_preparacao_dados.ipynb")
EXPERIMENTO = notebook("02_experimento_completo.ipynb")


def celula_com(nb, trecho):
    return next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and trecho in "".join(c["source"]))


def carregar_funcoes(nb):
    ns = dict(np=np, pd=pd, Path=Path, math=math, BATCH=16, json=json, hashlib=hashlib,
              zipfile=zipfile, os=__import__("os"))
    for c in nb["cells"]:
        if c["cell_type"] != "code" or "".join(c["source"]).lstrip().startswith("%"):
            continue
        arvore = ast.parse("".join(c["source"]))
        funcoes = [n for n in arvore.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=funcoes, type_ignores=[]), "<funcoes-notebook>", "exec"), ns)
    return ns


class Notebooks(unittest.TestCase):
    def setUp(self):
        self.a = carregar_funcoes(PREPARACAO)
        self.b = carregar_funcoes(EXPERIMENTO)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)

    def registro(self, conf=(0.9, 0.8), caixas=None, gt=None):
        return dict(pred_conf=np.array(conf),
                    pred_xyxy=np.array(caixas if caixas is not None else [[0, 0, 10, 10]] * len(conf)).reshape(-1, 4),
                    gt_xyxy=np.array(gt if gt is not None else [[0, 0, 10, 10]]).reshape(-1, 4))

    def test_sintaxe_de_todas_as_celulas(self):
        for nb in (PREPARACAO, EXPERIMENTO):
            ids = [c["id"] for c in nb["cells"]]
            self.assertEqual(len(ids), len(set(ids)))
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code":
                    codigo = "".join(c["source"])
                    compile(codigo, f"celula-{i}", "exec")
                    self.assertEqual(c["outputs"], [])
                    self.assertIsNone(c["execution_count"])

    def test_conversao_coco_yolo_e_leitura(self):
        linha = self.a["linha_yolo"]({"category_id": 1, "bbox": [10, 20, 30, 40]}, 100, 200)
        p = self.raiz / "label.txt"
        p.write_text(linha)
        for ns in (self.a, self.b):
            np.testing.assert_allclose(ns["ler_rotulos_yolo"](p, 100, 200), [[10, 20, 40, 60]])

    def test_conversao_recusa_caixas_invalidas(self):
        for box in ([0, 0, -1, 1], [90, 0, 20, 10], [float("nan"), 0, 1, 1]):
            with self.assertRaises(ValueError):
                self.a["linha_yolo"]({"category_id": 1, "bbox": box}, 100, 100)

    def test_rotulo_ausente_nao_vira_imagem_vazia(self):
        for ns in (self.a, self.b):
            with self.assertRaises(FileNotFoundError):
                ns["ler_rotulos_yolo"](self.raiz / "ausente.txt", 100, 100)

    def test_rotulo_vazio_e_valido(self):
        p = self.raiz / "vazio.txt"
        p.write_text("\n")
        for ns in (self.a, self.b):
            self.assertEqual(ns["ler_rotulos_yolo"](p, 100, 100).shape, (0, 4))

    def test_rotulo_malformado_nao_e_ignorado(self):
        p = self.raiz / "label.txt"
        for linha in ("0 0.5", "1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.9 0.5 0.5 0.2"):
            p.write_text(linha)
            for ns in (self.a, self.b):
                with self.assertRaises(ValueError):
                    ns["ler_rotulos_yolo"](p, 100, 100)

    def test_zip_recusa_escapar_do_destino(self):
        p = self.raiz / "dados.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("../fora.txt", "invalido")
        with self.assertRaises(ValueError):
            self.a["extrair_zip_seguro"](p, self.raiz / "extraido")
        self.assertFalse((self.raiz / "fora.txt").exists())

    def test_iou_geometria(self):
        r = self.b["iou_matriz"]([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]])
        np.testing.assert_array_equal(r, [[1, 0]])
        self.assertEqual(self.b["iou_matriz"]([], [[0, 0, 10, 10]]).shape, (0, 1))

    def test_deteccao_duplicada_e_falso_positivo(self):
        regs = [self.registro()]
        aval = self.b["avaliar"](regs, 0.5)
        met = self.b["metricas_no_limiar"](regs, aval, 0.8)
        self.assertEqual(met["recall"], 1)
        self.assertEqual(met["fppi"], 1)
        self.assertEqual(met["precisao"], 0.5)

    def test_empate_de_confianca_e_indivisivel(self):
        aval = self.b["avaliar"]([self.registro(conf=(0.8, 0.8))], 0.5)
        self.assertEqual(len(aval["curva"]), 1)
        self.assertEqual(aval["curva"].iloc[0].fppi, 1)

    def test_escolhe_maior_limiar_que_satisfaz_requisito(self):
        curva = pd.DataFrame(dict(limiar=[.9, .8, .7], recall=[.85, .9, 1], fppi=[0, .5, 1]))
        p = self.b["escolher_limiar"](curva, .9, 1)
        self.assertTrue(p["atingiu_alvo"])
        self.assertEqual(p["limiar"], .8)

    def test_melhor_recall_dentro_do_teto_se_alvo_falha(self):
        curva = pd.DataFrame(dict(limiar=[.9, .8, .7], recall=[.5, .6, .8], fppi=[0, .5, 2]))
        p = self.b["escolher_limiar"](curva, .9, 1)
        self.assertFalse(p["atingiu_alvo"])
        self.assertEqual(p["limiar"], .8)

    def test_sem_ponto_viavel_rejeita_todas_deteccoes_inclusive_conf_um(self):
        regs = [self.registro(conf=(1.0,), caixas=[[20, 20, 30, 30]])]
        aval = self.b["avaliar"](regs, .5)
        p = self.b["escolher_limiar"](aval["curva"], .9, 0)
        self.assertGreater(p["limiar"], 1)
        met = self.b["metricas_no_limiar"](regs, aval, p["limiar"])
        self.assertEqual(met["fppi"], 0)
        self.assertEqual(met["recall"], 0)
        self.assertFalse(p["atingiu_alvo"])

    def test_sem_deteccoes_preserva_falsos_negativos(self):
        regs = [self.registro(conf=())]
        aval = self.b["avaliar"](regs, .5)
        p = self.b["escolher_limiar"](aval["curva"], .9, 1)
        met = self.b["metricas_no_limiar"](regs, aval, p["limiar"])
        self.assertEqual(met["pessoas_nao_encontradas"], 1)
        self.assertEqual(met["fppi"], 0)

    def test_imagem_sem_pessoas_conta_alarmes(self):
        regs = [self.registro(conf=(.9,), gt=[])]
        met = self.b["metricas_no_limiar"](regs, self.b["avaliar"](regs, .5), .5)
        self.assertEqual(met["fppi"], 1)
        self.assertEqual(met["recall"], 0)

    def test_registro_identico_preserva_data_original(self):
        p = self.raiz / "protocolo.json"
        gravar = self.b["gravar_registro_unico"]
        gravar(p, {"data": "ontem", "batch": 16})
        original = p.read_bytes()
        gravar(p, {"data": "hoje", "batch": 16}, ignorar=("data",))
        self.assertEqual(p.read_bytes(), original)

    def test_mudanca_de_configuracao_nao_sobrescreve_registro(self):
        p = self.raiz / "especificacao.json"
        gravar = self.b["gravar_registro_unico"]
        gravar(p, {"cfg_treino": {"batch": 16, "lr0": .001}})
        original = p.read_bytes()
        with self.assertRaises(RuntimeError):
            gravar(p, {"cfg_treino": {"batch": 8, "lr0": .001}})
        self.assertEqual(p.read_bytes(), original)

    def test_modelo_antigo_nao_recebe_protocolo_novo(self):
        (self.raiz / "modelo_congelado.pt").write_bytes(b"modelo antigo")
        checkpoint = self.raiz / "best.pt"
        checkpoint.write_bytes(b"modelo atual")
        ns = dict(self.b, EXEC=self.raiz, CHECKPOINT=checkpoint,
                  SHA_CHECKPOINT_VALIDADO=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  hashlib=hashlib)
        with self.assertRaisesRegex(RuntimeError, "outro modelo"):
            exec(celula_com(EXPERIMENTO, 'MODELO_CONGELADO = EXEC'), ns)
        self.assertFalse((self.raiz / "protocolo_congelado.json").exists())

    def estado_teste(self):
        modelo = self.raiz / "modelo.pt"
        modelo.write_bytes(b"modelo de teste unitario")
        spec = self.raiz / "especificacao.json"
        spec.write_text('{}')
        protocolo = dict(checkpoint_sha256=hashlib.sha256(modelo.read_bytes()).hexdigest(),
                         especificacao_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
                         imgsz=1280, batch=16, iou_nms=.7, max_det=300, conf_min=.001,
                         confianca_do_artigo=.5, ponto_de_operacao=dict(limiar=.8, iou_match=.5))
        arq = self.raiz / "protocolo.json"
        arq.write_text(json.dumps(protocolo))
        chamadas = []
        ns = dict(self.b, ARQ_PROTOCOLO=arq, MODELO_CONGELADO=modelo, ARQ_ESPECIFICACAO=spec,
                  IMGSZ=1280, BATCH=16, IOU_NMS=.7, MAX_DET=300, CONF_MIN=.001,
                  EXEC=self.raiz, agora=lambda: "agora", hashlib=hashlib,
                  # Valores em memória conflitantes não devem determinar o teste.
                  ponto={"limiar": .1}, IOU_MATCH=.99, CONF_PAPER=.2,
                  catalogo=pd.DataFrame({"split": ["test"], "caminho": ["imagem.jpg"]}),
                  coletar_predicoes=lambda *args: [self.registro(conf=(.7,))],
                  metricas_ultralytics=lambda *args, **kwargs: chamadas.append(kwargs) or {})
        return ns, chamadas

    def test_avaliacao_usa_limiar_e_iou_do_arquivo_congelado(self):
        ns, chamadas = self.estado_teste()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(celula_com(EXPERIMENTO, 'protocolo = json.loads(ARQ_PROTOCOLO'), ns)
        resultado = json.loads((self.raiz / "resultado_teste.json").read_text())
        self.assertEqual(resultado["missao"]["confianca"], .8)
        self.assertEqual(resultado["missao"]["recall"], 0)
        self.assertEqual(ns["aval_teste"]["iou_min"], .5)
        self.assertEqual(chamadas[0]["conf"], .5)
        self.assertEqual(chamadas[1]["conf"], .001)

    def test_avaliacao_recusa_parametro_alterado_antes_da_inferencia(self):
        ns, chamadas = self.estado_teste()
        ns["IMGSZ"] = 640
        with self.assertRaises(RuntimeError):
            exec(celula_com(EXPERIMENTO, 'protocolo = json.loads(ARQ_PROTOCOLO'), ns)
        self.assertEqual(chamadas, [])

    def test_avaliacao_recusa_modelo_modificado_antes_da_inferencia(self):
        ns, chamadas = self.estado_teste()
        ns["MODELO_CONGELADO"].write_bytes(b"outros pesos")
        with self.assertRaises(AssertionError):
            exec(celula_com(EXPERIMENTO, 'protocolo = json.loads(ARQ_PROTOCOLO'), ns)
        self.assertEqual(chamadas, [])


if __name__ == "__main__":
    unittest.main()
