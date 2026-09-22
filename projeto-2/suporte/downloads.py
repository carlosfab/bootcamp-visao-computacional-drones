"""Transferência única com um arquivo parcial controlado pelo projeto."""
from pathlib import Path


def recusar_temporarios_gdown(destino):
    """Detecta sobras das versões que passavam um caminho ao gdown 5.2.0."""
    destino = Path(destino)
    parcial = destino.with_name(destino.name + ".part")
    antigos = sorted(
        p for p in destino.parent.iterdir()
        if p != parcial and p.name.startswith(parcial.name) and p.name.endswith(".part")
    )
    if antigos:
        caminhos = "\n".join(str(p) for p in antigos)
        raise RuntimeError(
            "Há temporários de um download anterior; nenhuma transferência foi iniciada. "
            "Confira a causa da interrupção e remova manualmente estes arquivos antes "
            f"de tentar novamente:\n{caminhos}"
        )


def transferir_gdown(parcial, **origem):
    """Grava diretamente no .part exclusivo, inclusive quando há Ctrl+C."""
    import gdown

    parcial = Path(parcial)
    try:
        # Com um objeto arquivo, gdown não cria outro temporário com nome aleatório.
        with parcial.open("xb") as saida:
            resultado = gdown.download(
                output=saida, quiet=False, use_cookies=False, resume=False,
                log_messages={"output": f"Destino parcial: {parcial}\n"}, **origem,
            )
        if resultado is None:
            raise RuntimeError("O servidor não forneceu o arquivo solicitado.")
    except (Exception, KeyboardInterrupt) as exc:
        mensagem = (
            f"Transferência interrompida. Parcial preservado: {parcial}. "
            "Confira a causa e remova esse arquivo manualmente antes de um novo download. "
            f"Detalhe: {exc}"
        )
        if isinstance(exc, KeyboardInterrupt):
            raise KeyboardInterrupt(mensagem) from exc
        raise RuntimeError(mensagem) from exc
