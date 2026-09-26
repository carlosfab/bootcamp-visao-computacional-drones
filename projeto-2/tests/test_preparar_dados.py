"""Verificação offline de aquisição, extração e integridade dos dados."""
import copy
import hashlib
import importlib.util
import io
import json
import stat
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HELPER = Path(__file__).resolve().parents[1] / "suporte" / "preparar_dados.py"
sys.path.insert(0, str(HELPER.parent))
SPEC = importlib.util.spec_from_file_location("preparar_dados_teste", HELPER)
dados = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dados)


class PrepararDadosTest(unittest.TestCase):
    def setUp(self):
        self.temporario = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporario.cleanup)
        self.raiz = Path(self.temporario.name).resolve() / "projeto"
        self.raiz.mkdir()
        self.download = Mock(side_effect=AssertionError("A rede não deveria ser acionada"))
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"gdown": SimpleNamespace(download=self.download)}).start()

    def arquivo(self, relativo, conteudo):
        caminho = self.raiz / relativo
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(conteudo)
        return {
            "destino": relativo,
            "nome_original": caminho.name,
            "bytes": len(conteudo),
            "sha256": hashlib.sha256(conteudo).hexdigest(),
            "google_drive_id": "apenas-fixture-offline",
            "url": "https://example.invalid/fixture",
        }

    def rgbt(self):
        anotacoes = []
        membros = {}
        for particao, nome in (("train", "00000.jpg"), ("val", "00001.jpg")):
            texto = json.dumps({"images": [{"file_name": nome}]}).encode()
            anotacoes.append(self.arquivo(
                f"dados/anotacoes-oficiais-extras/{particao}_thermal.json", texto,
            ))
            for sensor in ("visible", "thermal"):
                membros[f"{particao}/{sensor}/{nome}"] = f"{particao}:{sensor}".encode()
            membros[f"{particao}/annotation/{Path(nome).stem}.xml"] = b"<annotation/>"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as pacote:
            for nome, conteudo in membros.items():
                pacote.writestr(nome, conteudo)
        fonte = {
            "anotacoes": anotacoes,
            "arquivo": self.arquivo("dados/archives/RGBTDronePerson.zip", buffer.getvalue()),
        }
        patch.object(dados, "_fontes", return_value={"rgbtdroneperson": fonte}).start()
        return fonte, membros

    def test_extracao_rejeita_travessia_e_caminho_absoluto(self):
        nomes = ("../escape", str(self.raiz.parent / "escape"), "pasta/../../escape", "a\\escape")
        for nome in nomes:
            with self.subTest(nome=nome):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as pacote:
                    pacote.writestr(nome, b"conteudo")
                with zipfile.ZipFile(buffer) as pacote:
                    with self.assertRaises(ValueError):
                        dados._extrair_membro(pacote, pacote.infolist()[0], self.raiz, nome)
        self.assertFalse((self.raiz.parent / "escape").exists())
        self.assertEqual(list(self.raiz.iterdir()), [])

    def test_extracao_rejeita_link_simbolico_no_zip(self):
        info = zipfile.ZipInfo("atalho")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as pacote:
            pacote.writestr(info, "../fora")
        with zipfile.ZipFile(buffer) as pacote:
            with self.assertRaisesRegex(ValueError, "Tipo de membro"):
                dados._extrair_membro(pacote, pacote.infolist()[0], self.raiz, "atalho")
        self.assertFalse((self.raiz / "atalho").exists())

    def test_extracao_rejeita_destino_que_atravessa_link_local(self):
        (self.raiz / "atalho").symlink_to(self.raiz.parent, target_is_directory=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as pacote:
            pacote.writestr("ok.jpg", b"conteudo")
        with zipfile.ZipFile(buffer) as pacote:
            with self.assertRaises(ValueError):
                dados._extrair_membro(pacote, pacote.infolist()[0], self.raiz, "atalho/escape")
        self.assertFalse((self.raiz.parent / "escape").exists())

    def test_sha_divergente_nao_sobrescreve_nem_baixa(self):
        especificacao = self.arquivo("modelos/peso.pth", b"original")
        caminho = self.raiz / especificacao["destino"]
        caminho.write_bytes(b"alterado")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            dados._baixar(self.raiz, especificacao)
        self.assertEqual(caminho.read_bytes(), b"alterado")
        self.download.assert_not_called()

    def test_partial_incompleto_nao_inicia_novo_download(self):
        especificacao = self.arquivo("dados/pacote.bin", b"arquivo completo")
        (self.raiz / especificacao["destino"]).unlink()
        parcial = self.raiz / (especificacao["destino"] + ".part")
        parcial.write_bytes(b"incompleto")
        with self.assertRaisesRegex(RuntimeError, "parcial incompleto"):
            dados._baixar(self.raiz, especificacao)
        self.assertEqual(parcial.read_bytes(), b"incompleto")
        self.download.assert_not_called()

    def test_partial_completo_e_validado_sem_rede(self):
        especificacao = self.arquivo("dados/pacote.bin", b"arquivo completo")
        caminho = self.raiz / especificacao["destino"]
        parcial = caminho.with_name(caminho.name + ".part")
        caminho.rename(parcial)
        self.assertEqual(dados._baixar(self.raiz, especificacao), caminho)
        self.assertFalse(parcial.exists())
        self.download.assert_not_called()

    def test_quota_interrompe_apos_uma_tentativa(self):
        especificacao = self.arquivo("dados/pacote.bin", b"arquivo completo")
        (self.raiz / especificacao["destino"]).unlink()
        self.download.side_effect = RuntimeError("Quota exceeded")
        with self.assertRaisesRegex(RuntimeError, "quota excedida"):
            dados._baixar(self.raiz, especificacao)
        self.assertEqual(self.download.call_count, 1)

    def test_rgbt_sem_recibo_confere_zip_e_arquivos_antes_de_registrar(self):
        fonte, membros = self.rgbt()
        for nome, conteudo in membros.items():
            self.arquivo("dados/" + nome, conteudo)
        baixar = dados._baixar
        with patch.object(dados, "_baixar", wraps=baixar) as espiao:
            resultado = dados.preparar_rgbtdroneperson(self.raiz)
        self.assertIn(
            fonte["arquivo"], [chamada.args[1] for chamada in espiao.call_args_list],
        )
        recibo = json.loads((resultado / "recibo-rgbtdroneperson.json").read_text())
        self.assertEqual(recibo["arquivo_sha256"], fonte["arquivo"]["sha256"])
        self.assertEqual({m["nome"] for m in recibo["membros"]}, set(membros))
        self.assertFalse((resultado / "recibo-rgbtdroneperson.json.part").exists())
        self.download.assert_not_called()

    def test_rgbt_cache_com_recibo_dispensa_zip_mas_confere_crc(self):
        fonte, _ = self.rgbt()
        dados.preparar_rgbtdroneperson(self.raiz)
        (self.raiz / fonte["arquivo"]["destino"]).unlink()
        resultado = dados.preparar_rgbtdroneperson(self.raiz)
        self.assertEqual(resultado, self.raiz / "dados")
        self.download.assert_not_called()

    def test_rgbt_cache_corrompido_mesmo_tamanho_e_preservado(self):
        self.rgbt()
        dados.preparar_rgbtdroneperson(self.raiz)
        for relativo in ("train/visible/00000.jpg", "val/annotation/00001.xml"):
            caminho = self.raiz / "dados" / relativo
            original = caminho.read_bytes()
            corrompido = bytes([original[0] ^ 1]) + original[1:]
            caminho.write_bytes(corrompido)
            with self.subTest(relativo=relativo):
                with self.assertRaisesRegex(ValueError, "CRC32"):
                    dados.preparar_rgbtdroneperson(self.raiz)
                self.assertEqual(caminho.read_bytes(), corrompido)
            caminho.write_bytes(original)
        self.download.assert_not_called()

    def test_rgbt_recibo_incompleto_nao_valida_cache(self):
        self.rgbt()
        dados.preparar_rgbtdroneperson(self.raiz)
        caminho = self.raiz / "dados/recibo-rgbtdroneperson.json"
        recibo = json.loads(caminho.read_text())
        recibo["membros"].pop()
        caminho.write_text(json.dumps(recibo))
        with self.assertRaisesRegex(ValueError, "Recibo inválido"):
            dados.preparar_rgbtdroneperson(self.raiz)
        self.download.assert_not_called()

    def test_vtuav_reutiliza_os_178_membros_sem_acionar_rede(self):
        fontes = copy.deepcopy(dados._fontes())
        membros = fontes["vtuav"]["membros"]
        self.assertEqual(len(membros), 178)
        self.assertEqual(len({m["frame"] for m in membros}), 89)
        for membro in membros:
            conteudo = f"{membro['frame']}:{membro['modalidade']}".encode()
            self.arquivo(membro["destino"], conteudo)
            membro["bytes"] = len(conteudo)
            membro["sha256"] = hashlib.sha256(conteudo).hexdigest()
            membro["crc32"] = zlib.crc32(conteudo)
        with patch.object(dados, "_fontes", return_value=fontes):
            with patch.object(dados, "_baixar", side_effect=AssertionError("Não baixar ZIP")) as baixar:
                resultado = dados.preparar_vtuav(self.raiz)
        self.assertEqual(resultado, self.raiz / fontes["vtuav"]["destino_sequencia"])
        baixar.assert_not_called()
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
