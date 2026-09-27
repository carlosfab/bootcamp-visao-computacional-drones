#!/usr/bin/env python3
"""Gera as fontes dos notebooks. Execução e inspeção são etapas separadas."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/carlosfab/bootcamp-visao-computacional-drones.git"
BRANCH = "codex/projeto-4-gado"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    source = dedent(text).strip() + "\n"
    assert len(source.splitlines()) <= 12, source
    return {"cell_type": "code", "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}


def opening(filename, title, objective, gpu=False):
    cells = [md(f'''
    <div style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
      <img src="https://sigmoidal.ai/wp-content/uploads/2024/09/Academia-Sigmoidal-Light.png" alt="Academia Sigmoidal" width="250">
      <h1 style="color: #007bff; font-size: 24px;">Bootcamp de Visão Computacional com Drones</h1>
      <h3 style="color: #343a40; font-size: 20px;">Projeto 4: {title}</h3>
      <p style="color: #6c757d; font-size: 14px;"><strong>Instrutor:</strong> Carlos Melo, MSc.</p>
    </div>
    '''), md(f'''
    # {title}

    {objective}

    Este notebook é independente: os dados necessários acompanham o repositório. No Colab, selecione **Copiar para o Drive** e execute as células na ordem. {'Selecione uma GPU em Ambiente de execução > Alterar tipo de ambiente de execução.' if gpu else 'A CPU é suficiente para esta exploração.'}

    [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/{BRANCH}/projeto-4/{filename})
    '''), md('''
    ## Ambiente

    A primeira célula obtém os arquivos da versão do projeto e instala as dependências. O PyTorch fornecido pelo ambiente será preservado; sua versão será registrada. Se já importou bibliotecas antes desta instalação, reinicie o ambiente e execute novamente desde o início.

    Os arquivos de trabalho ficam em `dados/`, `pesos/` e `resultados/`. As pastas de métricas e figuras têm nomes fixos e seus arquivos são atualizados ao reexecutar. Antes de repetir o experimento ou encerrar o Colab, baixe o ZIP seguindo as instruções ao final.
    '''), code(f'''
    from pathlib import Path
    import os
    import subprocess
    import sys
    if not Path("suporte").is_dir():
        destino = Path("/content/bootcamp-gado")
        if not destino.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", "{BRANCH}",
                            "{REPO}", str(destino)], check=True)
        subprocess.run(["git", "-C", str(destino), "sparse-checkout", "set", "projeto-4"], check=True)
        os.chdir(destino / "projeto-4")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    '''), code('''
    import json
    import hashlib
    import platform
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from IPython.display import display
    from suporte.preparar_dados import obter_dados
    from suporte.experimento import inventario, ler_caixas, desenhar
    plt.rcParams.update({"figure.figsize": (9, 5), "font.size": 12,
                         "font.family": "sans-serif", "axes.grid": False})
    ''')]
    return cells


def environment(folder, models=True):
    cells = [md('''
    A semente controla parte da aleatoriedade, mas versões, hardware e operações numéricas também influenciam o resultado. O registro abaixo identifica o ambiente desta execução, sem prometer identidade bit a bit entre CPU e GPU.
    ''')]
    if models:
        cells.append(code('''
        import torch
        from ultralytics import YOLO
        from importlib.metadata import version
        from suporte.experimento import inferir, calibrar, salvar_resultado
        from suporte.contagem import evaluate_predictions
        DISPOSITIVO = 0 if torch.cuda.is_available() else "cpu"
        print("Dispositivo:", DISPOSITIVO)
        '''))
    else:
        cells.append(code('from importlib.metadata import version'))
    cells.append(code(f'''
    SAIDA = Path("resultados/{folder}")
    SAIDA.mkdir(parents=True, exist_ok=True)
    pacotes = ["numpy", "pandas", "matplotlib", "Pillow", "ultralytics", "torch"]
    ambiente = {{nome: version(nome) for nome in pacotes}}
    ambiente["python"] = platform.python_version()
    ambiente["sistema"] = platform.platform()
    {'ambiente["dispositivo"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"' if models else ''}
    (SAIDA / "ambiente.json").write_text(json.dumps(ambiente, indent=2))
    display(ambiente)
    '''))
    return cells


def data_cells():
    return [md('''
    ## Dados e partições

    O recorte didático contém **300 imagens** de bovinos a pasto: 180 de treino, 60 de validação e 60 de teste. As imagens derivam do ICAERUS v2, com lado maior limitado a 2048 pixels e anotações YOLO de uma classe, `cow`. A atribuição e a licença CC BY 4.0 acompanham os dados.

    Treino e validação usam Mauron e Jalogny em datas separadas. O teste usa Derval, uma fazenda ausente do desenvolvimento. Datas e voos não atravessam partições, mas os animais não têm identificadores individuais. A avaliação mede contagem **por imagem**, não um censo de indivíduos únicos.
    '''), code('''
    RAIZ = obter_dados()
    tabela = inventario(RAIZ)
    manifesto = json.loads((RAIZ / "manifesto.json").read_text())
    treino = tabela[tabela["split"] == "train"].copy()
    validacao = tabela[tabela["split"] == "val"].copy()
    teste = tabela[tabela["split"] == "test"].copy()
    assert [len(treino), len(validacao), len(teste)] == [180, 60, 60]
    print("Imagens de treino, validação e teste:", len(treino), len(validacao), len(teste))
    ''')]


def backup_instructions():
    return md('''
    ## Salvar a execução

    A pasta de um treinamento concluído é protegida contra reutilização; uma nova tentativa exige outro `name`. Já as pastas de métricas, configurações e figuras têm nomes fixos e são atualizadas ao reexecutar os notebooks. Baixe os resultados **antes de repetir** ou encerrar o runtime.

    No Colab, copie o trecho opcional abaixo para uma nova célula e execute-o quando quiser baixar a execução. O ZIP inclui `resultados/` inteiro, os pesos treinados que estiverem nessa pasta e o manifesto. O checkpoint de referência do notebook de recortes continua disponível nos arquivos do projeto e é identificado pelo hash da configuração.

    ```python
    import shutil
    from datetime import datetime
    from google.colab import files
    shutil.copy("assets/manifesto.json", "resultados/manifesto.json")
    nome = "resultados-projeto4-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    arquivo = shutil.make_archive(nome, "zip", "resultados")
    files.download(arquivo)
    ```

    Salve também este notebook com suas saídas em **Arquivo > Salvar** na cópia do Drive, ou use **Arquivo > Fazer download > Fazer download do .ipynb**. A cópia do notebook não salva automaticamente os arquivos temporários do runtime. Localmente, compacte `resultados/` ou use as mesmas linhas sem importar `google.colab` e sem chamar `files.download`.
    ''')


def gallery(folder, rows="previsoes_teste", threshold="limiar"):
    return [md('''
    ## Examinar erros concretos

    A galeria seleciona as imagens com mais falsos positivos e falsos negativos, usando o limiar já congelado. **Verde** representa a anotação; **vermelho tracejado**, a previsão. Uma contagem correta pode esconder uma omissão compensada por uma caixa falsa.

    Depois desta inspeção, registre hipóteses de falha. Uma alteração motivada pelo teste passa a ser exploração e exigiria outro conjunto independente para uma nova avaliação final.
    '''), code(f'''
    por_imagem = pd.read_csv(SAIDA / "{folder}" / "per_image.csv")
    por_imagem["falhas"] = por_imagem["fp"] + por_imagem["fn"]
    casos = por_imagem.sort_values(["falhas", "image_id"], ascending=[False, True]).head(4)
    display(casos[["image_id", "gt_count", "pred_count", "fp", "fn"]])
    indice = {{linha["image_id"]: linha for linha in {rows}}}
    (SAIDA / "galeria").mkdir(exist_ok=True)
    '''), code(f'''
    for numero, image_id in enumerate(casos["image_id"]):
        linha = indice[image_id]
        figura = desenhar(linha["image"], linha["gt_boxes"], linha["pred_boxes"],
                          linha["scores"], {threshold})
        figura.savefig(SAIDA / "galeria" / f"caso_{{numero}}.png", dpi=120)
        plt.show()
        plt.close(figura)
    ''')]


def notebook00():
    name = "00_dados_e_problema.ipynb"
    cells = opening(name, "Quantos bovinos aparecem nesta imagem?",
        "Vamos auditar os dados antes de usar um detector. O objetivo é entender as anotações, a escala dos animais e a separação entre desenvolvimento e teste.")
    cells += environment("dados", False) + data_cells()
    cells += [md('''
    ## O teste permanece reservado

    O inventário verifica que cada imagem tem seu arquivo de anotação. Para explorar distribuições, selecionar exemplos e formular hipóteses, usaremos somente **treino e validação**. As caixas de teste serão usadas posteriormente, com as configurações já fixadas.

    Um arquivo de anotação vazio significa imagem negativa. Um arquivo ausente é erro de integridade e não deve virar silenciosamente uma contagem zero.
    '''), code('''
    desenvolvimento = tabela[tabela["split"].isin(["train", "val"])].copy()
    resumo = desenvolvimento.groupby("split").agg(
        imagens=("image_id", "size"), caixas=("count", "sum"),
        negativas=("count", lambda x: int((x == 0).sum())))
    display(resumo)
    resumo.to_csv(SAIDA / "resumo_desenvolvimento.csv")
    '''), md('''
    **Imagens negativas** avaliam a capacidade de não inventar animais onde não há bovinos anotados. O treino foi enriquecido com imagens positivas; as proporções de treino e validação não precisam coincidir. Reportar apenas uma média sobre todas as imagens pode esconder comportamento ruim no subconjunto positivo.
    '''), code('''
    for split, grupo in desenvolvimento.groupby("split"):
        plt.hist(grupo["count"], bins=np.arange(0, desenvolvimento["count"].max() + 2),
                 alpha=0.6, label=split)
    plt.xlabel("Bovinos anotados por imagem")
    plt.ylabel("Imagens")
    plt.legend()
    plt.show()
    '''), md('''
    ## Separação por captura

    Fotografias de um mesmo voo podem ser muito parecidas. Conferiremos os grupos registrados no manifesto e a política de partição, sem inspecionar as imagens do teste. Mesmo com datas diferentes, treino e validação podem mostrar os mesmos animais; o experimento não testa identificação individual.
    '''), code('''
    amostras = pd.DataFrame(manifesto["samples"])
    display(amostras.groupby(["split", "farm"]).size().rename("imagens").to_frame())
    grupos = amostras.groupby(["farm", "flight"])["split"].nunique()
    assert grupos.max() == 1, "Um voo aparece em mais de uma partição."
    assert set(amostras.loc[amostras["split"] == "test", "farm"]) == {"Derval"}
    assert "Derval" not in set(amostras.loc[amostras["split"] != "test", "farm"])
    print("Voos separados; Derval reservada para o teste.")
    '''), md('''
    ## Ler uma anotação

    Cada linha YOLO contém `classe centro_x centro_y largura altura`, com coordenadas normalizadas. Multiplicar as coordenadas horizontais pela largura da imagem e as verticais pela altura recupera pixels. Para desenhar uma caixa, usamos seus cantos `x_min, y_min, x_max, y_max`.
    '''), code('''
    exemplo = treino[treino["count"] > 0].sort_values("image_id").iloc[0]
    linhas = Path(exemplo["label"]).read_text().splitlines()
    print("Primeira anotação:", linhas[0])
    caixas = ler_caixas(exemplo["label"], int(exemplo["width"]), int(exemplo["height"]))
    print("Primeira caixa em pixels:", caixas[0].round(1))
    print("Total de caixas:", len(caixas))
    '''), code('''
    figura = desenhar(exemplo["image"], caixas, [], [], 0.25)
    plt.title(f"{len(caixas)} bovinos anotados")
    plt.gca().get_legend().remove()
    figura.savefig(SAIDA / "referencia_anotada.png", dpi=120)
    plt.show()
    plt.close(figura)
    '''), md('''
    As caixas são a **referência anotada**, que também pode conter ambiguidades. Registre casos suspeitos sem alterar silenciosamente os arquivos. Uma correção de rótulos deve produzir uma nova versão dos dados e uma nova avaliação das configurações comparadas.

    ## O tamanho do animal na entrada da rede

    O tamanho equivalente de uma caixa é a raiz quadrada de sua área, em pixels. O inventário calcula a mediana desses tamanhos por imagem; não confunda essa estatística com a mediana de todas as caixas do conjunto.

    Ao reduzir o lado maior da foto para 640 pixels, cada animal também encolhe. A estimativa abaixo considera o redimensionamento proporcional, antes do preenchimento usado pelo detector.
    '''), code('''
    desenvolvimento["fator_640"] = 640 / desenvolvimento[["width", "height"]].max(axis=1)
    desenvolvimento["lado_640"] = desenvolvimento["median_box_side"] * desenvolvimento["fator_640"]
    colunas = ["width", "height", "median_box_side", "lado_640"]
    display(desenvolvimento[colunas].describe().round(2))
    desenvolvimento.to_csv(SAIDA / "inventario_desenvolvimento.csv", index=False)
    '''), code('''
    plt.hist(desenvolvimento["median_box_side"].dropna(), bins=20, alpha=0.6, label="Imagem")
    plt.hist(desenvolvimento["lado_640"].dropna(), bins=20, alpha=0.6, label="Entrada 640")
    plt.xlabel("Tamanho mediano das caixas por imagem, em pixels")
    plt.ylabel("Imagens")
    plt.legend()
    plt.show()
    '''), md('''
    Para observar a perda de detalhe, selecionamos uma imagem positiva da validação com animais pequenos. Exibir uma versão reduzida em um painel grande apenas amplia os pixels restantes; não recupera detalhe descartado.
    '''), code('''
    pequeno = validacao[validacao["count"] > 0].sort_values("median_box_side").iloc[0]
    foto = Image.open(pequeno["image"]).convert("RGB")
    reduzida = foto.copy()
    reduzida.thumbnail((640, 640))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for eixo, imagem, titulo in zip(axes, [foto, reduzida], ["Imagem disponível", "Lado maior 640"]):
        eixo.imshow(imagem)
        eixo.set_title(titulo)
        eixo.axis("off")
    plt.show()
    '''), md('''
    ## Comparar a mesma região em detalhe

    As duas fotografias inteiras ocupam quase o mesmo espaço na tela, o que pode esconder a perda de detalhe. Vamos ampliar a região do primeiro bovino anotado neste exemplo de validação, incluindo uma margem e respeitando os limites da imagem.

    O recorte é definido na imagem disponível. Para localizar a mesma região na versão reduzida, multiplicamos suas coordenadas pela proporção entre as dimensões das duas imagens. Arredondamos os limites para cobrir pixels inteiros e mantemos o mesmo campo de visão nos dois painéis.
    '''), code('''
    caixa_zoom = ler_caixas(pequeno["label"], foto.width, foto.height)[0]
    centro = (caixa_zoom[:2] + caixa_zoom[2:]) / 2
    raio = max(32, 2 * float(np.max(caixa_zoom[2:] - caixa_zoom[:2])))
    inicio_zoom = np.maximum(0, np.floor(centro - raio)).astype(int)
    fim_zoom = np.minimum(foto.size, np.ceil(centro + raio)).astype(int)
    escala = np.array(reduzida.size) / np.array(foto.size)
    inicio_red = np.floor(inicio_zoom * escala).astype(int)
    fim_red = np.minimum(reduzida.size, np.ceil(fim_zoom * escala)).astype(int)
    '''), code('''
    zoom_original = foto.crop((*inicio_zoom, *fim_zoom))
    zoom_reduzido = reduzida.crop((*inicio_red, *fim_red))
    campo_original = [inicio_zoom[0], fim_zoom[0], fim_zoom[1], inicio_zoom[1]]
    campo_reduzido = [inicio_red[0] / escala[0], fim_red[0] / escala[0],
                     fim_red[1] / escala[1], inicio_red[1] / escala[1]]
    print("Pixels no recorte disponível:", zoom_original.size)
    print("Pixels no recorte reduzido:", zoom_reduzido.size)
    '''), code('''
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    paineis = [(zoom_original, campo_original, "Imagem disponível"),
               (zoom_reduzido, campo_reduzido, "Após redução para 640")]
    for eixo, (imagem, campo, titulo) in zip(axes, paineis):
        eixo.imshow(imagem, extent=campo, interpolation="nearest")
        eixo.set_xlim(inicio_zoom[0], fim_zoom[0])
        eixo.set_ylim(fim_zoom[1], inicio_zoom[1])
        eixo.set_title(titulo)
        eixo.axis("off")
    fig.savefig(SAIDA / "zoom_resolucao.png", dpi=120)
    plt.show()
    '''), md('''
    A exibição usa o vizinho mais próximo para deixar os pixels visíveis, sem suavizar a ampliação. Os dois painéis mostram a mesma região, mas o recorte reduzido dispõe de menos pixels para representar o bovino. Aumentar o painel não recupera a informação descartada.
    '''), md('''
    ## Antes de treinar

    Registre uma hipótese sobre as dificuldades do detector: tamanho aparente, sombra, fundo ou proximidade entre bovinos. Use exemplos de treino ou validação como evidência. Não atribua mudanças de escala à altitude sem metadados que sustentem essa interpretação.

    No próximo notebook, verificaremos se um detector genérico já produz uma contagem útil. A hipótese deve poder ser rejeitada pelos resultados.
    ''')]
    return name, cells, False


def notebook01():
    name = "01_baseline_e_contagem.ipynb"
    cells = opening(name, "Baseline e contagem por imagem",
        "Vamos medir o desempenho de um YOLO11n pré-treinado em COCO, sem atualizar os pesos. Escolheremos o limiar na validação e avaliaremos o teste com essa decisão congelada.", True)
    cells += environment("baseline") + data_cells()
    cells += [md('''
    ## Um detector genérico como referência

    O **baseline** estabelece o que o modelo já consegue fazer. O identificador numérico de `cow` no COCO não precisa ser zero: a função de inferência seleciona a classe pelo nome e remapeia a saída para a classe única deste conjunto. Os pesos iniciais serão baixados pela biblioteca, sem chave de API.
    '''), code('''
    modelo = YOLO("yolo11n.pt")
    classes_bovino = {i: nome for i, nome in modelo.names.items() if nome in {"cow", "cattle"}}
    assert classes_bovino, "Classe de bovino não encontrada."
    display(classes_bovino)
    hash_peso = hashlib.sha256(Path("yolo11n.pt").read_bytes()).hexdigest()
    print("SHA-256 dos pesos:", hash_peso)
    '''), md(r'''
    ## Contagem correta não garante detecção correta

    Com $N$ imagens, o erro absoluto médio é $\operatorname{MAE}=\frac{1}{N}\sum_i |\hat{y}^{(i)}-y^{(i)}|$, em bovinos por imagem. O viés usa a diferença com sinal: negativo indica subcontagem e positivo, supercontagem.

    Para avaliar localização, associamos cada previsão a no máximo uma anotação, com **IoU de pelo menos 0,5**. Precisão, recall e F1 agregam TP, FP e FN sobre todas as imagens. São métricas em um ponto de operação, não AP ou mAP.

    No exemplo sintético abaixo, uma caixa correta e outra deslocada produzem a contagem certa, mas deixam um falso positivo e um falso negativo.
    '''), code('''
    caso = {"image_id": "exemplo", "gt_boxes": [[0, 0, 10, 10], [20, 0, 30, 10]],
            "pred_boxes": [[0, 0, 10, 10], [40, 0, 50, 10]], "scores": [0.9, 0.8]}
    display(evaluate_predictions([caso], conf_threshold=0.25)["summary"])
    '''), md('''
    ## Calibrar somente na validação

    A inferência guarda candidatos com confiança a partir de 0,05 e entrada de 640 pixels. Assim, podemos avaliar vários limiares sem executar novamente o detector. A regra é **maior F1 na validação**; em empate, menor MAE e depois maior limiar.

    Essa escolha procura equilibrar localização e contagem. Ela não garante o melhor resultado em outra fazenda, e a confiança do detector não é uma probabilidade calibrada de acerto.
    '''), code('''
    previsoes_val = inferir(modelo, validacao, device=DISPOSITIVO)
    limiar, calibracao = calibrar(previsoes_val)
    calibracao.to_csv(SAIDA / "calibracao.csv", index=False)
    display(calibracao[["threshold", "precision", "recall", "f1", "mae"]])
    print("Limiar escolhido:", limiar)
    '''), code('''
    plt.plot(calibracao["threshold"], calibracao["precision"], label="Precisão")
    plt.plot(calibracao["threshold"], calibracao["recall"], label="Recall")
    plt.plot(calibracao["threshold"], calibracao["f1"], label="F1")
    plt.xlabel("Limiar de confiança")
    plt.legend()
    plt.show()
    '''), md('''
    Antes de abrir o teste, salvamos as escolhas do experimento. A biblioteca aplica sua supressão de caixas por imagem; o caminho de inferência será o mesmo na comparação posterior com fine-tuning. O manifesto identifica os dados utilizados.
    '''), code('''
    configuracao = {"modelo": "yolo11n.pt", "weights_sha256": hash_peso,
                    "imgsz": 640, "conf": limiar, "iou_match": 0.5,
                    "selecao": "F1 val; empate por menor MAE e maior confiança",
                    "manifest_sha256": hashlib.sha256((RAIZ / "manifesto.json").read_bytes()).hexdigest()}
    (SAIDA / "configuracao.json").write_text(json.dumps(configuracao, indent=2))
    resumo_val = salvar_resultado(SAIDA / "val", previsoes_val, limiar)
    '''), md('''
    ## Avaliação congelada em Derval

    O teste contém uma fazenda ausente do desenvolvimento. Isso oferece um caso de mudança de domínio, sem garantir generalização para todas as fazendas. Imagens de um mesmo voo ainda são correlacionadas; sessenta imagens não equivalem a sessenta fazendas independentes.
    '''), code('''
    previsoes_teste = inferir(modelo, teste, device=DISPOSITIVO)
    resumo = salvar_resultado(SAIDA / "test", previsoes_teste, limiar)
    chaves = ["n_images", "precision", "recall", "f1", "mae", "rmse", "bias", "exact_accuracy"]
    display(pd.Series({chave: resumo[chave] for chave in chaves}, name="Teste"))
    display(resumo["by_presence"])
    display(resumo["timing"])
    '''), md('''
    A taxa de contagem exata considera imagens positivas e negativas. Confira também as métricas separadas: muitas imagens vazias podem tornar a média agregada favorável mesmo quando bovinos são omitidos. Nos negativos, a taxa de imagens com falso positivo mede quantas cenas vazias receberam pelo menos uma caixa.

    O tempo inclui a chamada ao modelo e seu pós-processamento, mas exclui a leitura das imagens e anotações. A primeira chamada pode conter aquecimento; essas medidas são descritivas desta execução, não um benchmark rigoroso.
    ''')]
    cells += gallery("test")
    cells += [md('''
    ## Conclusão do baseline

    Registre o principal tipo de erro, o MAE no teste e uma limitação desta avaliação. Não ajuste o limiar para melhorar as imagens da galeria. No fine-tuning, a pergunta será se a adaptação do mesmo modelo ao domínio modifica esse comportamento.
    ''')]
    return name, cells, True


def notebook02():
    name = "02_fine_tuning.ipynb"
    cells = opening(name, "Adaptar o detector às imagens aéreas",
        "Vamos ajustar o YOLO11n com o treino, selecionar o checkpoint pela validação e avaliar a contagem com o mesmo protocolo do baseline. O treinamento curto é uma hipótese experimental; uma melhoria não está garantida.", True)
    cells += environment("fine_tuning") + data_cells()
    cells += [md('''
    ## Configuração antes do treinamento

    Usaremos 25 épocas, batch de 8 imagens, entrada de 640 pixels e semente 42. A GPU é recomendada; a CPU pode executar, mas seu tempo não representa o percurso planejado para aula. Não instalamos outra versão de PyTorch sobre a fornecida pelo Colab.

    O diretório de treino deve ser novo. Para investigar outra configuração, escolha outro `name` e registre a mudança antes de consultar o teste. Isso protege a pasta de treino; a pasta fixa `resultados/fine_tuning` de métricas e figuras será atualizada. Baixe o ZIP da execução anterior antes de reexecutar desde o início. A semente não garante resultados idênticos entre diferentes ambientes.
    '''), code('''
    modelo = YOLO("yolo11n.pt")
    cfg = dict(data=str(RAIZ.relative_to(Path.cwd()) / "data.yaml"), epochs=25, batch=8, imgsz=640,
               seed=42, workers=2, deterministic=True, cache=False, plots=False,
               device=DISPOSITIVO, project="resultados/treino", name="gado", exist_ok=False)
    assert not Path(cfg["project"], cfg["name"]).exists(), "Use outro nome para uma nova execução."
    (SAIDA / "configuracao_treino.json").write_text(json.dumps(cfg, indent=2))
    display(cfg)
    '''), md('''
    **Fine-tuning** parte dos pesos genéricos e os atualiza com as imagens do domínio. A validação acompanha o treinamento e determina `best.pt`; o teste não participa da seleção. Os argumentos completos usados pela biblioteca também ficam registrados em `args.yaml` dentro da execução.

    Não interrompa a célula para escolher uma época que parece boa no teste. Se o runtime cair, trate os arquivos existentes como um treino possivelmente incompleto e preserve-os para diagnóstico.
    '''), code('''
    from time import perf_counter
    inicio = perf_counter()
    modelo.train(**cfg)
    segundos_treino = perf_counter() - inicio
    pasta_treino = Path(modelo.trainer.save_dir)
    melhor = Path(modelo.trainer.best)
    assert melhor.is_file(), "O treinamento não produziu best.pt."
    print("Treino concluído em segundos:", round(segundos_treino, 1))
    print("Checkpoint:", melhor)
    '''), md('''
    ## Ler as curvas de treinamento

    A perda de localização de treino e a de validação ajudam a investigar o ajuste. Queda da perda de treino sem melhora na validação pode indicar que o modelo está se adaptando demais à amostra. As curvas não autorizam escolher parâmetros pelo teste.
    '''), code('''
    historico = pd.read_csv(pasta_treino / "results.csv")
    historico.columns = historico.columns.str.strip()
    plt.plot(historico["epoch"], historico["train/box_loss"], label="Treino")
    plt.plot(historico["epoch"], historico["val/box_loss"], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Perda de localização")
    plt.legend()
    plt.show()
    '''), md('''
    O registro de conclusão associa o checkpoint aos bytes dos pesos e ao ambiente. Essa identidade permite distinguir seu treinamento de um peso de referência distribuído com o projeto. O arquivo só é gravado depois do retorno normal de `train()`.
    '''), code('''
    conclusao = {"weights": str(melhor.resolve().relative_to(Path.cwd())), "weights_sha256": hashlib.sha256(melhor.read_bytes()).hexdigest(),
                 "epochs_completed": len(historico), "seconds": segundos_treino,
                 "checkpoint_selection": "best.pt selecionado pela validação durante train()",
                 "initial_weights_sha256": hashlib.sha256(Path("yolo11n.pt").read_bytes()).hexdigest(),
                 "manifest_sha256": hashlib.sha256((RAIZ / "manifesto.json").read_bytes()).hexdigest()}
    (SAIDA / "treinamento_concluido.json").write_text(json.dumps(conclusao, indent=2))
    display(conclusao)
    ajustado = YOLO(str(melhor))
    display(ajustado.names)
    '''), md('''
    ## Calibrar o modelo ajustado

    A escala das confianças pode mudar após o treinamento. Escolheremos o limiar novamente na validação, pela mesma regra do baseline: maior F1; em empate, menor MAE e maior limiar. Entrada e regra de associação permanecem iguais.
    '''), code('''
    previsoes_val = inferir(ajustado, validacao, device=DISPOSITIVO)
    limiar, calibracao = calibrar(previsoes_val)
    calibracao.to_csv(SAIDA / "calibracao.csv", index=False)
    display(calibracao[["threshold", "precision", "recall", "f1", "mae"]])
    print("Limiar congelado:", limiar)
    salvar_resultado(SAIDA / "val", previsoes_val, limiar)
    '''), md('''
    Salvamos a configuração de inferência antes da avaliação final. Reutilizar o teste entre os métodos planejados permite comparação direta; mudar o treinamento depois de examinar esse teste seria um novo ciclo de desenvolvimento.
    '''), code('''
    avaliacao = {"weights_sha256": conclusao["weights_sha256"], "conf": limiar,
                 "imgsz": 640, "iou_match": 0.5, "tiled": False,
                 "selecao": "F1 val; empate por menor MAE e maior confiança"}
    (SAIDA / "configuracao_avaliacao.json").write_text(json.dumps(avaliacao, indent=2))
    previsoes_teste = inferir(ajustado, teste, device=DISPOSITIVO)
    resumo = salvar_resultado(SAIDA / "test", previsoes_teste, limiar)
    chaves = ["n_images", "precision", "recall", "f1", "mae", "rmse", "bias", "exact_accuracy"]
    display(pd.Series({chave: resumo[chave] for chave in chaves}, name="Fine-tuning"))
    display(resumo["by_presence"])
    '''), md('''
    ## Comparar com a referência genérica

    Se o baseline foi executado neste mesmo diretório, a tabela abaixo reúne seus resultados. Caso contrário, o notebook continua independente e apresenta apenas o treino atual; execute o notebook do baseline e reúna os arquivos ao preparar a entrega.

    Compare as mesmas imagens, versões e regras. Um F1 maior não implica necessariamente um MAE menor, pois omissões e falsos positivos podem se compensar na contagem.

    Este é o primeiro experimento: **pesos iniciais versus pesos ajustados**, com arquitetura e inferência mantidas. Identifique cada linha do relatório pelo hash dos pesos. No notebook seguinte, o segundo experimento usa um checkpoint de referência em imagem inteira e recortes; o peso de referência pode ser diferente do seu treino. Não atribua uma diferença entre esses checkpoints apenas aos recortes.
    '''), code('''
    comparacao = {"fine_tuning": {chave: resumo[chave] for chave in chaves}}
    arquivo_baseline = Path("resultados/baseline/test/resumo.json")
    if arquivo_baseline.exists():
        cfg_baseline = json.loads(Path("resultados/baseline/configuracao.json").read_text())
        assert cfg_baseline["manifest_sha256"] == conclusao["manifest_sha256"], "Baseline usa outros dados."
        baseline = json.loads(arquivo_baseline.read_text())
        comparacao["baseline"] = {chave: baseline[chave] for chave in chaves}
    tabela_comparacao = pd.DataFrame.from_dict(comparacao, orient="index")
    tabela_comparacao.to_csv(SAIDA / "comparacao.csv")
    display(tabela_comparacao)
    ''')]
    cells += gallery("test")
    cells += [md('''
    ## Interpretar o ajuste

    Identifique o que mudou e o que permaneceu difícil. Distinga o efeito sobre imagens positivas e negativas, e explique por que sua recomendação pode favorecer o baseline mesmo depois de um treinamento válido.

    Preserve `best.pt`, `args.yaml`, `results.csv`, a configuração e o registro de ambiente. No próximo notebook, um checkpoint fixado permitirá estudar o efeito dos recortes sem depender de um treinamento anterior neste runtime.
    ''')]
    return name, cells, True


def notebook03():
    name = "03_objetos_pequenos.ipynb"
    cells = opening(name, "Objetos pequenos e inferência em recortes",
        "Vamos comparar imagem inteira e recortes sobrepostos usando exatamente o mesmo checkpoint. Avaliaremos detecção, contagem e tempo; os recortes podem melhorar, piorar ou apenas aumentar o custo.", True)
    cells += environment("recortes") + data_cells()
    cells += [md('''
    ## Um checkpoint fixado

    O pacote de pesos permite executar este notebook sem treinar antes. Os hashes verificam se usamos os mesmos bytes da distribuição, sem garantir qualidade preditiva. A avaliação será recalculada nesta sessão.

    O projeto contém dois experimentos pareados pelas imagens de teste. No primeiro, baseline e fine-tuning mantêm arquitetura e inferência e mudam os pesos. Neste segundo, **imagem inteira e recortes mantêm exatamente o mesmo checkpoint de referência**. Não substitua a linha de imagem inteira deste notebook pelo resultado de seu treino anterior: seus hashes podem ser diferentes.

    Para comparar seu próprio treinamento posteriormente, troque o checkpoint antes de calibrar e use outra pasta de resultados. Não misture pesos diferentes na comparação entre imagem inteira e recortes.
    '''), code('''
    import zipfile
    from suporte.contagem import tile_windows, predict_image, nms
    metadados_peso = json.loads(Path("assets/pesos-gado.json").read_text())
    pacote_peso = Path("assets/pesos-gado.zip")
    assert hashlib.sha256(pacote_peso.read_bytes()).hexdigest() == metadados_peso["archive_sha256"]
    with zipfile.ZipFile(pacote_peso) as pacote:
        assert "pesos/gado.pt" in pacote.namelist()
        Path("pesos").mkdir(exist_ok=True)
        Path("pesos/gado.pt").write_bytes(pacote.read("pesos/gado.pt"))
    PESO = Path("pesos/gado.pt")
    assert hashlib.sha256(PESO.read_bytes()).hexdigest() == metadados_peso["weights_sha256"]
    modelo = YOLO(str(PESO))
    '''), md('''
    ## Por que recortar?

    Uma imagem com lado maior de 2048 pixels perde detalhe ao ser reduzida para uma entrada de 640. Um recorte de 640 pixels conserva mais pixels por animal, mas limita o contexto. A sobreposição reduz o risco de cortar um bovino na borda e também produz previsões repetidas.

    O experimento fixa recortes de **640 pixels**, sobreposição nominal de **20%** e NMS global com **IoU de 0,5**. Nas bordas, a última janela é alinhada à imagem e a sobreposição real pode ser maior. Esses parâmetros são definidos antes de examinar o teste.
    '''), code('''
    pequeno = validacao[validacao["count"] > 0].sort_values("median_box_side").iloc[0]
    foto = np.asarray(Image.open(pequeno["image"]).convert("RGB"))
    janelas = tile_windows(foto.shape[1], foto.shape[0], tile_size=640, overlap=0.2)
    print("Dimensões:", foto.shape)
    print("Quantidade de recortes:", len(janelas))
    print("Primeiras janelas:", janelas[:3])
    '''), code('''
    from matplotlib.patches import Rectangle
    plt.figure(figsize=(12, 7))
    plt.imshow(foto)
    for x1, y1, x2, y2 in janelas:
        plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, edgecolor="orange", linewidth=1))
    plt.axis("off")
    plt.show()
    '''), md('''
    ## Voltar ao referencial original

    O detector devolve as caixas no referencial de cada recorte. Somar o deslocamento horizontal aos dois valores de `x` e o vertical aos dois valores de `y` recupera as coordenadas da imagem original. Só depois dessa transformação podemos comparar e deduplicar caixas de janelas diferentes.
    '''), code('''
    caixas, confiancas = [], []
    for x1, y1, x2, y2 in janelas:
        previsao = predict_image(modelo, foto[y1:y2, x1:x2], conf=0.05, device=DISPOSITIVO)
        caixas.append(previsao["boxes"] + np.array([x1, y1, x1, y1]))
        confiancas.append(previsao["scores"])
    caixas = np.concatenate(caixas)
    confiancas = np.concatenate(confiancas)
    mantidas = nms(caixas, confiancas, iou_threshold=0.5)
    print("Caixas antes e depois da NMS:", len(caixas), len(mantidas))
    '''), md('''
    A **supressão não máxima (NMS)** prioriza caixas de maior confiança e remove sobreposições acima do limiar. Ela reduz duplicatas, mas pode suprimir caixas de bovinos próximos ou manter fragmentos de um mesmo animal. Portanto, somar as contagens dos recortes é incorreto, e aplicar NMS não dispensa inspeção.

    O limiar 0,25 abaixo é apenas uma escolha de exibição da demonstração na validação. O limiar de avaliação será calibrado usando todas as imagens de validação.
    '''), code('''
    gt = ler_caixas(pequeno["label"], foto.shape[1], foto.shape[0])
    for titulo, indices in [("Antes da NMS", np.arange(len(caixas))), ("Depois da NMS", mantidas)]:
        print(titulo)
        figura = desenhar(foto, gt, caixas[indices], confiancas[indices], 0.25)
        plt.show()
        plt.close(figura)
    '''), md('''
    ## Comparação controlada na validação

    A única mudança de inferência entre os métodos é processar a imagem inteira ou os recortes e reuni-los por NMS. Cada método tem seu limiar escolhido pela mesma regra de validação: maior F1, depois menor MAE e maior limiar.

    Os recortes fazem várias chamadas ao detector por imagem. O tempo medido inclui inferência e pós-processamento, exclui leitura dos dados e pode incluir aquecimento. Execute ambos no mesmo dispositivo e evite interpretar pequenas diferenças como conclusivas.
    '''), code('''
    val_inteira = inferir(modelo, validacao, device=DISPOSITIVO, tiled=False)
    limiar_inteira, curva_inteira = calibrar(val_inteira)
    val_recortes = inferir(modelo, validacao, device=DISPOSITIVO, tiled=True)
    limiar_recortes, curva_recortes = calibrar(val_recortes)
    curva_inteira.to_csv(SAIDA / "calibracao_inteira.csv", index=False)
    curva_recortes.to_csv(SAIDA / "calibracao_recortes.csv", index=False)
    print("Limiar da imagem inteira:", limiar_inteira)
    print("Limiar dos recortes:", limiar_recortes)
    '''), code('''
    salvar_resultado(SAIDA / "val_inteira", val_inteira, limiar_inteira)
    salvar_resultado(SAIDA / "val_recortes", val_recortes, limiar_recortes)
    configuracao = {"weights_sha256": metadados_peso["weights_sha256"], "imgsz": 640,
                    "tile_size": 640, "overlap": 0.2, "global_nms_iou": 0.5,
                    "conf_inteira": limiar_inteira, "conf_recortes": limiar_recortes,
                    "iou_match": 0.5, "device": str(DISPOSITIVO),
                    "manifest_sha256": hashlib.sha256((RAIZ / "manifesto.json").read_bytes()).hexdigest()}
    (SAIDA / "configuracao.json").write_text(json.dumps(configuracao, indent=2))
    display(configuracao)
    '''), md('''
    ## Avaliar o teste com as decisões congeladas

    Agora aplicaremos os dois métodos às mesmas imagens de Derval. A tabela deve sustentar uma decisão entre benefício e custo, inclusive quando os recortes não ajudam. Nenhuma mudança de parâmetros será feita a partir deste resultado.
    '''), code('''
    teste_inteira = inferir(modelo, teste, device=DISPOSITIVO, tiled=False)
    teste_recortes = inferir(modelo, teste, device=DISPOSITIVO, tiled=True)
    resumo_inteira = salvar_resultado(SAIDA / "test_inteira", teste_inteira, limiar_inteira)
    resumo_recortes = salvar_resultado(SAIDA / "test_recortes", teste_recortes, limiar_recortes)
    chaves = ["n_images", "precision", "recall", "f1", "mae", "rmse", "bias", "exact_accuracy"]
    comparacao = {nome: {chave: resumo[chave] for chave in chaves}
                 for nome, resumo in [("inteira", resumo_inteira), ("recortes", resumo_recortes)]}
    for nome, resumo in [("inteira", resumo_inteira), ("recortes", resumo_recortes)]:
        comparacao[nome]["segundos_por_imagem"] = resumo["timing"]["mean_seconds"]
    comparacao = pd.DataFrame.from_dict(comparacao, orient="index")
    comparacao.to_csv(SAIDA / "comparacao.csv")
    display(comparacao)
    '''), md('''
    O custo depende da quantidade de janelas. Abaixo registramos essa quantidade para cada imagem e mostramos separadamente positivos e negativos. Taxa alta de contagem exata em cenas vazias não basta para concluir que o detector conta bem as cenas com gado.
    '''), code('''
    custo = teste[["image_id", "width", "height"]].copy()
    custo["recortes"] = [len(tile_windows(int(w), int(h)))
                         for w, h in zip(custo["width"], custo["height"])]
    custo.to_csv(SAIDA / "janelas_por_imagem.csv", index=False)
    display(custo["recortes"].describe())
    display({"inteira": resumo_inteira["by_presence"], "recortes": resumo_recortes["by_presence"]})
    ''')]
    cells += gallery("test_recortes", "teste_recortes", "limiar_recortes")
    cells += [md('''
    Compare as mesmas imagens selecionadas acima com a previsão na imagem inteira. Observe animais recuperados, caixas falsas e possíveis duplicatas ou supressões. Não pressuponha que todas essas falhas ocorrerão neste conjunto.
    '''), code('''
    indice_inteira = {linha["image_id"]: linha for linha in teste_inteira}
    for image_id in casos["image_id"]:
        linha = indice_inteira[image_id]
        figura = desenhar(linha["image"], linha["gt_boxes"], linha["pred_boxes"],
                          linha["scores"], limiar_inteira)
        plt.show()
        plt.close(figura)
    '''), md('''
    ## Uma recomendação baseada em evidências

    Recomende uma configuração para o cenário avaliado. Use MAE, viés, F1, erros em positivos e negativos, custo e pelo menos quatro exemplos para justificar. Identifique o hash dos pesos em cada linha da tabela final. O efeito dos recortes é medido pelo par deste notebook, com o mesmo checkpoint; diferenças para o treino do notebook anterior não podem ser atribuídas só ao tiling. Distinga o que os resultados demonstram do que permanece uma hipótese.

    Este experimento não resolve identidade entre fotografias, contagem de um rebanho inteiro ou funcionamento em qualquer fazenda. A próxima avaliação poderia reservar outros voos e fazendas, com referência independente e um protocolo de cobertura definido.
    ''')]
    return name, cells, True


def main():
    for factory in (notebook00, notebook01, notebook02, notebook03):
        name, cells, gpu = factory()
        cells.append(backup_instructions())
        metadata = {"colab": {"provenance": []},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "language_info": {"name": "python"}}
        if gpu:
            metadata["colab"]["gpuType"] = "T4"
            metadata["accelerator"] = "GPU"
        notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}
        for cell in cells:
            assert "—" not in cell["source"], name
        (ROOT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        print(name, len(cells), "células")


if __name__ == "__main__":
    main()
