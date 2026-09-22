#!/usr/bin/env python3
"""Prepare os ambientes locais dos notebooks, sem modificar o Python do sistema."""

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


PROJETO = Path(__file__).resolve().parents[1]
EDA = PROJETO / ".venv-eda"
QFDET = PROJETO / ".venv-qfdet-cpu"
FONTE = PROJETO / "codigo" / "qfdet"
REPOSITORIO = "https://github.com/NNNNerd/mmdet-rgbtdroneperson.git"
COMMIT = "a79210c283597294ded255028bb8fe16e1c241cd"


def executar(argumentos, env=None, capturar=False):
    argumentos = [str(valor) for valor in argumentos]
    print("\n$ " + shlex.join(argumentos), flush=True)
    resultado = subprocess.run(
        argumentos, cwd=PROJETO, env=env, check=True,
        text=True, capture_output=capturar,
    )
    return resultado.stdout.strip() if capturar else None


def localizar_uv():
    executavel = shutil.which("uv")
    if not executavel:
        candidato = Path.home() / ".local" / "bin" / "uv"
        if candidato.is_file() and os.access(candidato, os.X_OK):
            executavel = str(candidato)
    if not executavel:
        raise RuntimeError(
            "uv não encontrado. Instale-o conforme "
            "https://docs.astral.sh/uv/getting-started/installation/ "
            "e execute este comando novamente."
        )
    return executavel


def conferir_plataforma():
    sistema, arquitetura = platform.system(), platform.machine().lower()
    if arquitetura not in {"x86_64", "amd64"} or sistema not in {"Darwin", "Linux"}:
        raise RuntimeError(
            f"Plataforma {sistema}/{arquitetura} fora deste roteiro. "
            "A execução do curso foi validada em Mac Intel; o roteiro Linux x86_64 "
            "é experimental. Apple Silicon, Windows e Colab não foram validados."
        )
    if sistema == "Linux":
        print("Linux x86_64: roteiro experimental, ainda não validado de ponta a ponta.")
    return sistema


def preparar_venv(uv, pasta, versao):
    python = pasta / "bin" / "python"
    if pasta.exists() and not python.is_file():
        raise RuntimeError(f"{pasta.name} existe, mas não é um ambiente válido; nada foi apagado.")
    if not pasta.exists():
        executar([uv, "venv", "--no-project", "--python", versao, pasta])
    obtida = executar(
        [python, "-c", "import platform; print(platform.python_version())"], capturar=True
    )
    if obtida != versao:
        raise RuntimeError(
            f"{pasta.name} usa Python {obtida}; este roteiro exige {versao}. "
            "O ambiente existente foi preservado. Use uma cópia limpa do projeto "
            "ou mova esse ambiente manualmente antes de tentar novamente."
        )
    return python


def instalar_requisitos(uv, python, arquivo):
    executar([
        uv, "pip", "install", "--python", python,
        "--only-binary", ":all:", "-r", PROJETO / arquivo,
    ])


def preparar_eda(uv):
    python = preparar_venv(uv, EDA, "3.12.3")
    instalar_requisitos(uv, python, "requirements-eda.txt")
    executar([uv, "pip", "check", "--python", python])
    executar([
        python, "-m", "ipykernel", "install", "--prefix", EDA,
        "--name", "drones-projeto2-eda", "--display-name", "Projeto 2 - EDA",
    ])
    executar([python, "-c", "import numpy, pandas, matplotlib, PIL; print('Imports EDA: OK')"])
    print("\nEDA preparado. Abra o JupyterLab com:")
    print(shlex.join([str(python), "-m", "jupyterlab", str(PROJETO)]))


def preparar_fonte():
    if not shutil.which("git"):
        raise RuntimeError("Git não encontrado. Instale o Git antes de preparar QFDet.")
    if FONTE.exists():
        if not (FONTE / ".git").exists():
            raise RuntimeError(f"{FONTE} já existe e não é um clone Git; nada foi alterado.")
        commit = executar(["git", "-C", FONTE, "rev-parse", "HEAD"], capturar=True)
        alteracoes = executar(
            ["git", "-C", FONTE, "status", "--porcelain", "--untracked-files=no"],
            capturar=True,
        )
        if commit != COMMIT or alteracoes:
            raise RuntimeError(
                "O clone codigo/qfdet possui outra revisão ou alterações locais. "
                f"Este roteiro usa {COMMIT}. O clone foi preservado; "
                "revise suas alterações ou mova a pasta antes de tentar novamente."
            )
    else:
        FONTE.parent.mkdir(parents=True, exist_ok=True)
        executar(["git", "clone", "--no-checkout", REPOSITORIO, FONTE])
        executar(["git", "-C", FONTE, "checkout", "--detach", COMMIT])


def preparar_qfdet(uv, sistema):
    python_eda = EDA / "bin" / "python"
    if not python_eda.is_file():
        raise RuntimeError("Primeiro execute: python3 scripts/preparar_ambiente.py --eda")
    executar([python_eda, "-c", "import jupyterlab"])
    if sistema == "Darwin":
        if not Path("/usr/bin/clang++").is_file():
            raise RuntimeError("Instale as ferramentas de desenvolvimento da Apple (clang++).")
    elif not shutil.which("c++"):
        raise RuntimeError("Instale um compilador C++ antes de compilar MMCV (por exemplo, build-essential).")

    preparar_fonte()
    python = preparar_venv(uv, QFDET, "3.10.11")
    instalar_requisitos(uv, python, "requirements-build.txt")
    instalar_requisitos(uv, python, "requirements-qfdet.txt")
    if sistema == "Darwin":
        executar([uv, "pip", "install", "--python", python,
                  "--no-deps", "torch==1.13.1", "torchvision==0.14.1"])
    else:
        executar([uv, "pip", "install", "--python", python,
                  "--no-deps", "torch==1.13.1+cpu", "torchvision==0.14.1+cpu",
                  "--index-url", "https://download.pytorch.org/whl/cpu"])

    ambiente = os.environ.copy()
    ambiente.update({
        "MMCV_WITH_OPS": "1", "FORCE_CUDA": "0", "MAX_JOBS": "4",
        "PATH": str(QFDET / "bin") + os.pathsep + ambiente.get("PATH", ""),
    })
    if sistema == "Darwin":
        ambiente.update({
            "MACOSX_DEPLOYMENT_TARGET": "12.3",
            "CFLAGS": "-stdlib=libc++ -mmacosx-version-min=12.3 -Wno-invalid-specialization",
            "CXXFLAGS": "-mmacosx-version-min=12.3 -Wno-invalid-specialization",
            "CC": "/usr/bin/clang", "CXX": "/usr/bin/clang++",
        })
    print("\nMMCV pode levar alguns minutos para compilar; o pip reutiliza seu cache quando possível.")
    executar([
        python, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
        "mmcv-full==1.6.1",
    ], env=ambiente)
    executar([python, "-m", "pip", "check"])
    verificacao = """
import sys
sys.path.insert(0, sys.argv[1])
import torch, torchvision, mmcv, mmdet
from mmcv.ops import nms
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
assert torch.__version__.split('+')[0] == '1.13.1'
assert torchvision.__version__.split('+')[0] == '0.14.1'
assert mmcv.__version__ == '1.6.1'
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
scores = torch.tensor([0.9, 0.8])
_, indices = nms(boxes.cpu(), scores.cpu(), 0.5)
assert indices.tolist() == [0], indices
print('QFDet importado e NMS executado em CPU: OK')
print('torch:', torch.__version__, '| mmcv:', mmcv.__version__, '| mmdet:', mmdet.__version__)
"""
    executar([python, "-c", verificacao, FONTE])
    executar([
        python, "-m", "ipykernel", "install", "--prefix", EDA,
        "--name", "drones-projeto2-qfdet", "--display-name", "Projeto 2 - QFDet CPU",
    ])
    print("\nQFDet preparado. No JupyterLab do ambiente EDA, selecione o kernel Projeto 2 - QFDet CPU.")
    print("A preparação verifica imports e NMS; os notebooks executam a inferência com os pesos dos autores.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepara ambientes separados para EDA (00/01) e QFDet (02/03), somente em CPU.",
        epilog="Execute --eda primeiro. Instalações e kernels ficam dentro do projeto; não há treino.",
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--eda", action="store_true", help="Python 3.12.3, JupyterLab e notebooks 00/01")
    grupo.add_argument("--qfdet", action="store_true", help="Python 3.10.11, QFDet e notebooks 02/03")
    args = parser.parse_args()
    try:
        sistema = conferir_plataforma()
        uv = localizar_uv()
        if args.eda:
            preparar_eda(uv)
        else:
            preparar_qfdet(uv, sistema)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as erro:
        print(f"\nPreparação interrompida: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
