"""Downloads oficiais opcionais do Projeto 2, com verificação de integridade.

Exemplos (a partir de projeto-2):
    python suporte/preparar_dados.py --rgbtdroneperson --pesos rgbtdroneperson
    python suporte/preparar_dados.py --pesos vtuav
    python suporte/preparar_dados.py --vtuav

--vtuav baixa o ZIP oficial completo (~18,85 GB) somente se os 89 pares
selecionados ainda não estiverem disponíveis. Não há download ao importar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath

try:
    from .downloads import recusar_temporarios_gdown, transferir_gdown
except ImportError:
    from downloads import recusar_temporarios_gdown, transferir_gdown

FONTES = Path(__file__).with_name("fontes.json")
CHUNK = 1024 * 1024


def _fontes():
    return json.loads(FONTES.read_text(encoding="utf-8"))


def _projeto(projeto):
    raiz = Path(projeto).expanduser().resolve()
    raiz.mkdir(parents=True, exist_ok=True)
    return raiz


def _seguro(raiz, nome):
    """Rejeita caminhos absolutos, travessia e links dentro do destino."""
    if not nome or "\\" in nome or ":" in nome:
        raise ValueError(f"Caminho não permitido: {nome!r}")
    relativo = PurePosixPath(nome)
    if relativo.is_absolute() or ".." in relativo.parts:
        raise ValueError(f"Caminho não permitido: {nome!r}")
    destino = raiz.joinpath(*relativo.parts)
    try:
        destino.resolve().relative_to(raiz.resolve())
    except ValueError as exc:
        raise ValueError(f"Caminho fora do destino: {nome!r}") from exc
    atual = raiz
    for parte in relativo.parts:
        atual = atual / parte
        if atual.is_symlink():
            raise ValueError(f"Link simbólico não permitido: {atual}")
    return destino


def _conferir(arquivo, esperado):
    """Confere tamanho e, quando disponíveis, SHA-256 e CRC32."""
    if not arquivo.is_file():
        raise FileNotFoundError(f"Arquivo ausente: {arquivo}")
    if arquivo.stat().st_size != esperado["bytes"]:
        raise ValueError(f"Tamanho divergente: {arquivo}")
    sha = hashlib.sha256() if esperado.get("sha256") else None
    crc = 0
    if sha is not None or "crc32" in esperado:
        with arquivo.open("rb") as entrada:
            for bloco in iter(lambda: entrada.read(CHUNK), b""):
                if sha is not None:
                    sha.update(bloco)
                if "crc32" in esperado:
                    crc = zlib.crc32(bloco, crc)
    if sha is not None and sha.hexdigest() != esperado["sha256"]:
        raise ValueError(f"SHA-256 divergente: {arquivo}")
    if "crc32" in esperado and crc & 0xFFFFFFFF != esperado["crc32"]:
        raise ValueError(f"CRC32 divergente: {arquivo}")
    if esperado.get("nome_original", "").endswith(".zip") and not zipfile.is_zipfile(arquivo):
        raise ValueError(f"Conteúdo não é um ZIP válido: {arquivo}")
    return arquivo


def _parcial_valido(parcial, destino, esperado):
    """Reaproveita só um .part completo; nunca retoma uma transferência."""
    if not parcial.exists():
        return False
    try:
        _conferir(parcial, esperado)
    except (ValueError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Arquivo parcial incompleto ou inválido: {parcial}. "
            "A execução foi interrompida sem nova tentativa. Confira a causa "
            "do download anterior e remova esse .part manualmente antes de "
            "iniciar outra transferência."
        ) from exc
    parcial.replace(destino)
    return True


def _baixar(raiz, especificacao):
    destino = _seguro(raiz, especificacao["destino"])
    if destino.exists():
        return _conferir(destino, especificacao)
    destino.parent.mkdir(parents=True, exist_ok=True)
    recusar_temporarios_gdown(destino)
    parcial = _seguro(raiz, especificacao["destino"] + ".part")
    if _parcial_valido(parcial, destino, especificacao):
        return destino
    if shutil.disk_usage(destino.parent).free < especificacao["bytes"]:
        raise RuntimeError(
            f"Espaço insuficiente para {destino.name}: o download exige "
            f"{especificacao['bytes']:,} bytes livres, além da extração."
        )
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Instale a dependência: python -m pip install gdown") from exc
    print(f"Fonte oficial: {especificacao['url']}", flush=True)
    try:
        transferir_gdown(
            parcial,
            id=especificacao["google_drive_id"],
        )
        _conferir(parcial, especificacao)
    except Exception as exc:
        raise RuntimeError(
            f"Download interrompido: {destino.name}. Não houve retry nem "
            "troca de fonte. Se o Google Drive informar quota excedida ou "
            "acesso bloqueado, aguarde a liberação na fonte oficial. "
            f"Detalhe: {exc}"
        ) from exc
    parcial.replace(destino)
    return destino


def _info_segura(info):
    tipo = stat.S_IFMT(info.external_attr >> 16)
    if tipo not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Tipo de membro ZIP não permitido: {info.filename}")
    if info.flag_bits & 1:
        raise ValueError(f"Membro ZIP criptografado: {info.filename}")


def _extrair_membro(zipado, info, raiz, nome, esperado=None):
    _info_segura(info)
    _seguro(raiz, info.filename)
    destino = _seguro(raiz, nome)
    if info.is_dir():
        destino.mkdir(parents=True, exist_ok=True)
        return destino
    metadados = {"bytes": info.file_size, "crc32": info.CRC}
    if esperado is not None:
        if info.file_size != esperado["bytes"] or info.CRC != esperado["crc32"]:
            raise ValueError(f"Índice ZIP difere do manifesto: {info.filename}")
        metadados.update(esperado)
    if destino.exists():
        return _conferir(destino, metadados)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = _seguro(raiz, nome + ".part")
    if _parcial_valido(parcial, destino, metadados):
        return destino
    # ZipExtFile verifica CRC ao alcançar EOF; a conferência abaixo também
    # verifica o arquivo gravado e seu SHA-256 quando conhecido.
    with zipado.open(info) as entrada, parcial.open("xb") as saida:
        shutil.copyfileobj(entrada, saida, CHUNK)
    _conferir(parcial, metadados)
    parcial.replace(destino)
    return destino


def _indice_unico(zipado):
    indice = {}
    for info in zipado.infolist():
        if info.filename in indice:
            raise ValueError(f"Membro ZIP duplicado: {info.filename}")
        indice[info.filename] = info
    return indice


def _nomes_rgbt(dados):
    """Enumera todas as imagens e os XMLs correspondentes aos JSONs oficiais."""
    nomes = set()
    for particao in ("train", "val"):
        anotacoes = dados / "anotacoes-oficiais-extras" / f"{particao}_thermal.json"
        documento = json.loads(anotacoes.read_text(encoding="utf-8"))
        for registro in documento["images"]:
            for sensor in ("visible", "thermal"):
                nomes.add(f"{particao}/{sensor}/{registro['file_name']}")
            xml = PurePosixPath(registro["file_name"]).with_suffix(".xml")
            nomes.add(f"{particao}/annotation/{xml}")
    for nome in nomes:
        _seguro(dados, nome)
    return nomes


def _rgbt_disponivel(dados, fonte):
    """Só reutiliza o cache com recibo completo e CRC de todos os arquivos."""
    recibo = _seguro(dados, "recibo-rgbtdroneperson.json")
    if not recibo.exists():
        return False
    try:
        registro = json.loads(recibo.read_text(encoding="utf-8"))
        if registro["versao"] != 1 or registro["arquivo_sha256"] != fonte["arquivo"]["sha256"]:
            raise ValueError("Versão ou origem divergente")
        membros = registro["membros"]
        nomes = [membro["nome"] for membro in membros]
        if len(set(nomes)) != len(nomes) or set(nomes) != _nomes_rgbt(dados):
            raise ValueError("Lista de membros incompleta ou divergente")
        for membro in membros:
            if not isinstance(membro["bytes"], int) or membro["bytes"] < 0:
                raise ValueError("Tamanho inválido")
            if not isinstance(membro["crc32"], int) or not 0 <= membro["crc32"] <= 0xFFFFFFFF:
                raise ValueError("CRC inválido")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Recibo inválido: {recibo}: {exc}") from exc
    faltam = False
    for membro in membros:
        caminho = _seguro(dados, membro["nome"])
        if caminho.exists():
            _conferir(caminho, membro)
        else:
            faltam = True
    return not faltam


def _salvar_recibo_rgbt(dados, fonte, indice):
    membros = [
        {"nome": nome, "bytes": info.file_size, "crc32": info.CRC}
        for nome, info in sorted(indice.items()) if not info.is_dir()
    ]
    if {m["nome"] for m in membros} != _nomes_rgbt(dados):
        raise ValueError("O índice RGBTD não corresponde às imagens e XMLs esperados.")
    registro = {
        "versao": 1,
        "arquivo_sha256": fonte["arquivo"]["sha256"],
        "origem": "Metadados do ZIP oficial após conferir seu SHA-256 e extrair com CRC32.",
        "membros": membros,
    }
    recibo = _seguro(dados, "recibo-rgbtdroneperson.json")
    if recibo.exists():
        if json.loads(recibo.read_text(encoding="utf-8")) != registro:
            raise ValueError(f"Recibo divergente, preservado sem substituição: {recibo}")
        return
    parcial = _seguro(dados, "recibo-rgbtdroneperson.json.part")
    if parcial.exists():
        raise RuntimeError(f"Recibo parcial existente, preservado: {parcial}")
    with parcial.open("x", encoding="utf-8") as saida:
        json.dump(registro, saida, ensure_ascii=False, indent=2)
        saida.write("\n")
    parcial.replace(recibo)


def preparar_rgbtdroneperson(projeto):
    """Prepara imagens e os dois JSONs COCO; devolve a pasta dados."""
    raiz = _projeto(projeto)
    fonte = _fontes()["rgbtdroneperson"]
    for anotacao in fonte["anotacoes"]:
        _baixar(raiz, anotacao)
    dados = _seguro(raiz, "dados")
    if _rgbt_disponivel(dados, fonte):
        return dados
    arquivo = _baixar(raiz, fonte["arquivo"])
    with zipfile.ZipFile(arquivo) as zipado:
        indice = _indice_unico(zipado)
        # Valida todos os caminhos e tipos antes de extrair o primeiro membro.
        for info in indice.values():
            _info_segura(info)
            _seguro(dados, info.filename)
        for info in indice.values():
            _extrair_membro(zipado, info, dados, info.filename)
        _salvar_recibo_rgbt(dados, fonte, indice)
    if not _rgbt_disponivel(dados, fonte):
        raise RuntimeError("A extração terminou, mas faltam imagens do RGBTDronePerson.")
    return dados


def preparar_pesos(projeto, nome):
    """Baixa/reutiliza o checkpoint oficial e devolve seu caminho."""
    pesos = _fontes()["pesos"]
    if nome not in pesos:
        raise ValueError(f"Peso desconhecido: {nome!r}; escolha {', '.join(pesos)}.")
    return _baixar(_projeto(projeto), pesos[nome])


def preparar_vtuav(projeto):
    """Prepara exatamente os 89 pares da comparação e devolve a sequência."""
    raiz = _projeto(projeto)
    fonte = _fontes()["vtuav"]
    membros = fonte["membros"]
    faltam = False
    for membro in membros:
        destino = _seguro(raiz, membro["destino"])
        if destino.exists():
            _conferir(destino, membro)
        else:
            faltam = True
    sequencia = _seguro(raiz, fonte["destino_sequencia"])
    if not faltam:
        return sequencia
    print(
        "VTUAV: para obter os 89 pares, será usado o ZIP oficial completo "
        "(18,85 GB). Só os 178 JPEGs selecionados serão extraídos.",
        flush=True,
    )
    arquivo = _baixar(raiz, fonte["arquivo"])
    with zipfile.ZipFile(arquivo) as zipado:
        indice = _indice_unico(zipado)
        for membro in membros:
            if membro["nome_original"] not in indice:
                raise ValueError(f"Membro ausente no ZIP: {membro['nome_original']}")
            info = indice[membro["nome_original"]]
            _info_segura(info)
            _seguro(raiz, info.filename)
            if info.file_size != membro["bytes"] or info.CRC != membro["crc32"]:
                raise ValueError(f"Índice ZIP difere do manifesto: {info.filename}")
        for membro in membros:
            _extrair_membro(
                zipado, indice[membro["nome_original"]], raiz,
                membro["destino"], membro,
            )
    return sequencia


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projeto", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rgbtdroneperson", action="store_true", help="Imagens (~1,57 GB) e dois JSONs.")
    parser.add_argument("--vtuav", action="store_true", help="ZIP oficial (~18,85 GB), extraindo só 89 pares.")
    parser.add_argument("--pesos", action="append", choices=("rgbtdroneperson", "vtuav"), default=[])
    args = parser.parse_args(argv)
    if not (args.rgbtdroneperson or args.vtuav or args.pesos):
        parser.error("Selecione --rgbtdroneperson, --vtuav ou --pesos NOME.")
    try:
        if args.rgbtdroneperson:
            print(preparar_rgbtdroneperson(args.projeto))
        for nome in dict.fromkeys(args.pesos):
            print(preparar_pesos(args.projeto, nome))
        if args.vtuav:
            print(preparar_vtuav(args.projeto))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
