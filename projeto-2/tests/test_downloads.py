"""Regressões offline dos downloads, incluindo o transporte real do gdown."""
import contextlib
import hashlib
import importlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SUPORTE = Path(__file__).resolve().parents[1] / "suporte"
sys.path.insert(0, str(SUPORTE))
import preparar_dados as dados
import qfdet_video as video

try:
    import gdown as GDOWN_REAL
    TRANSPORTE_REAL = importlib.import_module("gdown.download")
except ImportError:
    GDOWN_REAL = TRANSPORTE_REAL = None


class CasosDownload:
    """O mesmo contrato é exercitado pelas duas entradas usadas nos notebooks."""

    def setUp(self):
        temporario = tempfile.TemporaryDirectory()
        self.addCleanup(temporario.cleanup)
        self.raiz = Path(temporario.name).resolve()
        self.destino = self.raiz / "modelos/peso.pth"
        self.destino.parent.mkdir()
        self.parcial = self.destino.with_name(self.destino.name + ".part")
        self.conteudo = b"checkpoint completo da fixture"
        self.sha256 = hashlib.sha256(self.conteudo).hexdigest()
        self.especificacao = {
            "destino": str(self.destino.relative_to(self.raiz)),
            "nome_original": self.destino.name,
            "bytes": len(self.conteudo),
            "sha256": self.sha256,
            "google_drive_id": "apenas-fixture-offline",
            "url": "https://example.invalid/peso.pth",
        }
        self.saidas = []

    def gravador(self, conteudo, erro=None):
        def baixar(**argumentos):
            saida = argumentos["output"]
            self.assertFalse(isinstance(saida, (str, Path)))
            self.assertEqual(Path(saida.name), self.parcial)
            self.assertFalse(argumentos["resume"])
            self.assertFalse(argumentos["use_cookies"])
            self.saidas.append(saida)
            saida.write(conteudo)
            if erro is not None:
                raise erro("Interrupção simulada após receber bytes")
            return saida
        return Mock(side_effect=baixar)

    def verificar_apenas_parcial(self, conteudo):
        self.assertFalse(self.destino.exists())
        self.assertEqual(self.parcial.read_bytes(), conteudo)
        self.assertEqual(list(self.destino.parent.iterdir()), [self.parcial])
        self.assertTrue(all(saida.closed for saida in self.saidas))

    def verificar_retry_bloqueado(self, download, conteudo):
        with self.assertRaisesRegex((RuntimeError, ValueError), "[Pp]arcial"):
            self.baixar()
        self.assertEqual(download.call_count, 1)
        self.verificar_apenas_parcial(conteudo)

    def test_sucesso_promove_arquivo_apos_fechar_e_validar(self):
        download = self.gravador(self.conteudo)
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            self.assertEqual(self.baixar(), self.destino)
            self.assertEqual(self.destino.read_bytes(), self.conteudo)
            self.assertEqual(list(self.destino.parent.iterdir()), [self.destino])
            self.assertTrue(self.saidas[0].closed)
            self.assertEqual(self.baixar(), self.destino)
        self.assertEqual(download.call_count, 1)

    def test_erro_preserva_bytes_e_bloqueia_nova_transferencia(self):
        parcial = self.conteudo[:7]
        download = self.gravador(parcial, RuntimeError)
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            with self.assertRaisesRegex(RuntimeError, "Interrupção simulada"):
                self.baixar()
            self.verificar_retry_bloqueado(download, parcial)

    def test_ctrl_c_preserva_bytes_fecha_arquivo_e_bloqueia_retry(self):
        parcial = self.conteudo[:7]
        download = self.gravador(parcial, KeyboardInterrupt)
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            with self.assertRaises(KeyboardInterrupt):
                self.baixar()
            self.verificar_retry_bloqueado(download, parcial)

    def test_sha_invalido_preserva_arquivo_e_bloqueia_retry(self):
        corrompido = b"X" + self.conteudo[1:]
        download = self.gravador(corrompido)
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            with self.assertRaisesRegex((RuntimeError, ValueError), "SHA-256"):
                self.baixar()
            self.verificar_retry_bloqueado(download, corrompido)

    def test_legado_aleatorio_e_preservado_e_bloqueia_antes_da_rede(self):
        legado = self.parcial.with_name(self.parcial.name + "abc12345.part")
        legado.write_bytes(b"download antigo interrompido")
        download = Mock(side_effect=AssertionError("Não deve acionar a rede"))
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            with self.assertRaises(RuntimeError) as erro:
                self.baixar()
        self.assertIn(str(legado), str(erro.exception))
        self.assertEqual(legado.read_bytes(), b"download antigo interrompido")
        self.assertEqual(list(self.destino.parent.iterdir()), [legado])
        download.assert_not_called()

    def test_legado_de_outro_destino_com_prefixo_semelhante_nao_bloqueia(self):
        vizinho = self.destino.with_name(self.destino.name + ".outro.part")
        vizinho.write_bytes(b"parcial de outro arquivo")
        download = self.gravador(self.conteudo)
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            self.assertEqual(self.baixar(), self.destino)
        self.assertEqual(vizinho.read_bytes(), b"parcial de outro arquivo")
        self.assertEqual(self.destino.read_bytes(), self.conteudo)
        self.assertEqual(download.call_count, 1)

    def test_cache_integro_dispensa_rede_mesmo_com_legado(self):
        self.destino.write_bytes(self.conteudo)
        legado = self.parcial.with_name(self.parcial.name + "abc12345.part")
        legado.write_bytes(b"download antigo")
        download = Mock(side_effect=AssertionError("Não deve acionar a rede"))
        with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=download)}):
            self.assertEqual(self.baixar(), self.destino)
        self.assertEqual(legado.read_bytes(), b"download antigo")
        download.assert_not_called()

    @unittest.skipUnless(GDOWN_REAL is not None, "gdown não está instalado")
    def test_gdown_real_interrompido_nao_cria_temporario_aleatorio(self):
        # A sessão HTTP é substituída: a escolha entre caminho e objeto
        # arquivo, a escrita e o tratamento do temporário são do gdown real.
        for tipo_erro in (RuntimeError, KeyboardInterrupt):
            with self.subTest(erro=tipo_erro.__name__):
                recebidos = self.conteudo[:7]

                class Resposta:
                    status_code = 200
                    headers = {
                        "Content-Type": "application/octet-stream",
                        "Content-Disposition": 'attachment; filename="peso.pth"',
                        "Content-Length": str(len(self.conteudo)),
                    }

                    def iter_content(self, chunk_size):
                        yield recebidos
                        raise tipo_erro("Conexão interrompida na fixture")

                sessao = Mock()
                sessao.get.return_value = Resposta()
                with (
                    patch.dict(sys.modules, {"gdown": GDOWN_REAL}),
                    patch.object(TRANSPORTE_REAL, "_get_session", return_value=(sessao, None)),
                    # Estes testes não usam processos; evita semáforos do tqdm.
                    patch.object(TRANSPORTE_REAL.tqdm.tqdm, "_lock", threading.RLock(), create=True),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(tipo_erro):
                        self.baixar()
                    self.verificar_apenas_parcial(recebidos)
                    with self.assertRaises((RuntimeError, ValueError)):
                        self.baixar()
                self.assertEqual(sessao.get.call_count, 1)
                sessao.close.assert_called_once()
                self.verificar_apenas_parcial(recebidos)
                self.parcial.unlink()


class DownloadDadosTest(CasosDownload, unittest.TestCase):
    def baixar(self):
        return dados._baixar(self.raiz, self.especificacao)


class DownloadPesoVideoTest(CasosDownload, unittest.TestCase):
    def baixar(self):
        return video.baixar_peso(self.destino, self.especificacao["url"], self.sha256)


if __name__ == "__main__":
    unittest.main()
