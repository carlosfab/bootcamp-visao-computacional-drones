#!/usr/bin/env python3
"""Gera somente o notebook 04; preserva os notebooks 00 a 03 executados."""
import ast
import hashlib
import json
from pathlib import Path

from gerar_notebooks import opening, environment, md, code, backup_instructions, gallery

ROOT = Path(__file__).resolve().parents[1]
NAME = "04_treino_com_recortes.ipynb"


def notebook04():
    cells = opening(NAME, "Treinar e inferir em escalas compatíveis",
        "Vamos investigar uma estratégia de treinamento com recortes e comparar o modelo ajustado com o YOLO11n genérico usando a mesma inferência em recortes. O experimento foi proposto após examinar Derval; sua avaliação nessa fazenda é exploratória.", True)
    cells += environment("escala_compativel")
    cells += [md('''
    ## Uma hipótese surgida da avaliação

    A avaliação anterior motivou esta pergunta: treinar reduzindo a fotografia inteira e depois inferir em recortes pode apresentar os bovinos ao modelo em escalas diferentes? A hipótese é que treinar com recortes preserve mais pixels por animal e torne a escala de treino mais próxima da inferência. Ela não prova a causa dos erros nem garante uma melhora.

    **Derval já foi observada** ao formular essa hipótese. Seus resultados abaixo são exploratórios, mesmo mantendo as mesmas 60 imagens. O desafio adicional usa 62 capturas de `Other_farms` que não haviam sido usadas para escolher estes parâmetros; não ajustaremos a estratégia depois de consultar seus resultados.

    `Other_farms` é uma pasta agregada, sem garantia de uma fazenda única ou independência geográfica. As 62 imagens são positivas: esse desafio mede erros em cenas com bovinos anotados e não permite estimar a taxa de alarmes em cenas vazias. Ele também não representa um censo de animais únicos.
    '''), md('''
    ## Preparar os dados sem recortar o teste para treino

    O conjunto base mantém 180 imagens de treino, 60 de validação e 60 de Derval. Apenas treino e validação fornecerão recortes para ajustar pesos e selecionar o checkpoint. A calibração de confiança continuará usando as **60 fotografias originais de validação**, com inferência em recortes e deduplicação global.

    Recortes da mesma fotografia herdam a partição e continuam correlacionados. Aumentar o número de arquivos não cria novas capturas independentes.
    '''), code('''
    from suporte.recortes_treino import preparar_recortes
    RAIZ = obter_dados()
    tabela = inventario(RAIZ)
    validacao = tabela[tabela["split"] == "val"].copy()
    derval = tabela[tabela["split"] == "test"].copy()
    assert [sum(tabela["split"] == "train"), len(validacao), len(derval)] == [180, 60, 60]
    RAIZ_RECORTES = preparar_recortes(RAIZ)
    manifesto_recortes = json.loads((RAIZ_RECORTES / "manifesto.json").read_text())
    display(pd.DataFrame.from_dict(manifesto_recortes["summary"], orient="index"))
    '''), md('''
    ## O que acontece nas bordas dos recortes?

    Usamos janelas de 640 pixels com sobreposição nominal de 20%, sem redimensionar seu conteúdo. As caixas são transladadas e limitadas à janela. Se qualquer caixa intersectante retiver menos de 50% de sua área, ou ficar com um lado menor que 2 pixels, **o recorte inteiro é descartado** para evitar ensinar fragmentos ambíguos como fundo.

    O treino mantém todos os recortes positivos válidos e até a mesma quantidade de negativos, selecionados por uma regra determinística. A validação mantém todos os recortes válidos. As contagens de recortes descartados, positivos, negativos e caixas ficam no manifesto; esse balanceamento não representa a prevalência natural de uma pastagem.
    '''), code('''
    amostras = [s for s in manifesto_recortes["samples"] if s["split"] == "train"]
    exemplo = next(s for s in amostras if (RAIZ_RECORTES / s["label"]).read_text().strip())
    imagem_recorte = Image.open(RAIZ_RECORTES / exemplo["image"]).convert("RGB")
    caixas = ler_caixas(RAIZ_RECORTES / exemplo["label"], *imagem_recorte.size)
    print("Imagem de origem:", exemplo["parent_id"])
    print("Janela na origem:", exemplo["window"])
    figura = desenhar(imagem_recorte, caixas, [], [], 0.25)
    plt.title(f"Recorte de treino: {len(caixas)} bovinos anotados")
    plt.gca().get_legend().remove()
    figura.savefig(SAIDA / "recorte_anotado.png", dpi=120)
    plt.show()
    plt.close(figura)
    '''), md('''
    Os manifestos associam arquivos, hashes, imagens de origem e partições. Guardá-los junto aos resultados permite verificar que o treino usou somente recortes derivados das partições autorizadas.
    '''), code('''
    import shutil
    for origem, nome in [(RAIZ / "manifesto.json", "manifesto_base.json"),
                         (RAIZ_RECORTES / "manifesto.json", "manifesto_recortes.json")]:
        shutil.copy(origem, SAIDA / nome)
    hash_base = hashlib.sha256((RAIZ / "manifesto.json").read_bytes()).hexdigest()
    hash_recortes = hashlib.sha256((RAIZ_RECORTES / "manifesto.json").read_bytes()).hexdigest()
    assert {s["split"] for s in manifesto_recortes["samples"]} == {"train", "val"}
    print("SHA-256 dos recortes:", hash_recortes)
    '''), md('''
    ## Fixar a estratégia de treinamento

    O ajuste parte do YOLO11n pré-treinado em COCO. Treinaremos por **20 épocas**, batch 16 e entrada 640, com semente 42, `mosaic=0` e `scale=0.25`. Desligar o mosaico evita combinar imagens em uma escala adicional; a variação de escala restante é moderada.

    Este experimento altera também épocas, batch e transformações em relação ao treino anterior. Portanto, avalia um conjunto de decisões; uma eventual melhora não isola causalmente o efeito dos recortes. A GPU é recomendada, sem estimativa de duração antes da medição.

    A pasta de treino deve ser nova. Se for repetir, baixe antes o ZIP e use outro `name`: a pasta fixa de métricas `resultados/escala_compativel` será atualizada.
    '''), code('''
    modelo = YOLO("yolo11n.pt")
    hash_inicial = hashlib.sha256(Path("yolo11n.pt").read_bytes()).hexdigest()
    cfg = dict(data=str(RAIZ_RECORTES.relative_to(Path.cwd()) / "data.yaml"),
               epochs=20, batch=16, imgsz=640, seed=42, workers=2,
               deterministic=True, cache=False, plots=False, mosaic=0, scale=0.25,
               device=DISPOSITIVO, project="resultados/treino", name="gado_recortes", exist_ok=False)
    assert not Path(cfg["project"], cfg["name"]).exists(), "Use outro name para repetir o treino."
    (SAIDA / "configuracao_treino.json").write_text(json.dumps(cfg, indent=2))
    display(cfg)
    '''), md('''
    A biblioteca acompanha a validação **em recortes** para selecionar `best.pt`. O teste não participa dessa seleção. Depois, o limiar será escolhido em fotografias completas da validação, exatamente pelo caminho de inferência usado na aplicação.

    Os argumentos completos ficam em `args.yaml`. Uma queda de runtime pode deixar arquivos parciais; a conclusão abaixo só será registrada após o retorno normal do treinamento.
    '''), code('''
    from time import perf_counter
    inicio = perf_counter()
    modelo.train(**cfg)
    segundos_treino = perf_counter() - inicio
    pasta_treino = Path(modelo.trainer.save_dir)
    melhor = Path(modelo.trainer.best)
    assert melhor.is_file(), "O treinamento não produziu best.pt."
    print("Tempo de treinamento em segundos:", round(segundos_treino, 1))
    '''), md('''
    ## Registrar o treinamento concluído

    A perda de localização mostra a evolução do ajuste nos recortes. Não compare sua escala numérica diretamente à perda do experimento anterior como prova de superioridade: os exemplos de treinamento mudaram. A avaliação em fotografias completas responderá à pergunta aplicada.
    '''), code('''
    historico = pd.read_csv(pasta_treino / "results.csv")
    historico.columns = historico.columns.str.strip()
    assert len(historico) == cfg["epochs"], "O histórico não contém as 20 épocas previstas."
    plt.plot(historico["epoch"], historico["train/box_loss"], label="Treino")
    plt.plot(historico["epoch"], historico["val/box_loss"], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Perda de localização")
    plt.legend()
    plt.show()
    '''), code('''
    hash_ajustado = hashlib.sha256(melhor.read_bytes()).hexdigest()
    conclusao = {"weights": str(melhor.resolve().relative_to(Path.cwd())),
                 "weights_sha256": hash_ajustado, "initial_weights_sha256": hash_inicial,
                 "epochs_completed": len(historico), "seconds": segundos_treino,
                 "manifest_sha256": hash_base, "crop_manifest_sha256": hash_recortes,
                 "checkpoint_selection": "best.pt pela validação em recortes durante train()"}
    (SAIDA / "treinamento_concluido.json").write_text(json.dumps(conclusao, indent=2))
    display(conclusao)
    ajustado = YOLO(str(melhor))
    coco = YOLO("yolo11n.pt")
    '''), md('''
    ## Calibrar dois modelos no mesmo percurso

    Compararemos o modelo ajustado e uma referência genérica COCO. **Ambos** processam as mesmas fotografias com recortes de 640 pixels, sobreposição de 20% e NMS global com IoU 0,5. Assim, o comparativo externo não mistura inferência em imagem inteira e recortes.

    A função de inferência seleciona bovinos pelo nome da classe e os remapeia para a classe única do conjunto. O índice de `cow` no COCO não precisa coincidir com o índice no modelo ajustado.

    Cada modelo recebe seu próprio limiar, escolhido nas 60 imagens originais de validação: maior F1, depois menor MAE e maior limiar. As previsões usam associação um a um com IoU mínimo de 0,5; precisão, recall e F1 são métricas nesse ponto de operação, não AP ou mAP.
    '''), code('''
    val_ajustado = inferir(ajustado, validacao, device=DISPOSITIVO, tiled=True)
    limiar_ajustado, curva_ajustado = calibrar(val_ajustado)
    val_coco = inferir(coco, validacao, device=DISPOSITIVO, tiled=True)
    limiar_coco, curva_coco = calibrar(val_coco)
    curva_ajustado.to_csv(SAIDA / "calibracao_ajustado.csv", index=False)
    curva_coco.to_csv(SAIDA / "calibracao_coco.csv", index=False)
    print("Limiar ajustado:", limiar_ajustado)
    print("Limiar COCO:", limiar_coco)
    '''), code('''
    resumo_val_ajustado = salvar_resultado(SAIDA / "val_ajustado", val_ajustado, limiar_ajustado)
    resumo_val_coco = salvar_resultado(SAIDA / "val_coco", val_coco, limiar_coco)
    chaves = ["n_images", "precision", "recall", "f1", "mae", "rmse", "bias", "exact_accuracy"]
    display(pd.DataFrame({"ajustado": {k: resumo_val_ajustado[k] for k in chaves},
                          "coco": {k: resumo_val_coco[k] for k in chaves}}).T)
    '''), md('''
    ## Congelar antes de avaliar

    O registro abaixo fixa pesos, limiares, dados e inferência. O hash do manifesto da reserva identifica seus bytes, sem abrir imagens nem usar seus rótulos para selecionar parâmetros. A partir daqui, qualquer alteração motivada por Derval ou pela reserva é um novo ciclo exploratório e exige uma nova reserva para avaliação independente.

    O desafio adicional tem 62 capturas ainda não usadas na escolha desta configuração. Essa condição de reserva não comprova independência de fazendas, animais ou voos em relação aos demais dados; o agrupamento `Other_farms` não fornece essa garantia.
    '''), code('''
    hash_reserva = hashlib.sha256(Path("assets/reserva-manifesto.json").read_bytes()).hexdigest()
    configuracao = {"adjusted_weights_sha256": hash_ajustado, "coco_weights_sha256": hash_inicial,
                    "base_manifest_sha256": hash_base, "crop_manifest_sha256": hash_recortes,
                    "reserve_manifest_sha256": hash_reserva, "imgsz": 640, "tile_size": 640,
                    "overlap": 0.2, "global_nms_iou": 0.5, "iou_match": 0.5,
                    "conf_ajustado": limiar_ajustado, "conf_coco": limiar_coco,
                    "threshold_selection": "F1 nas 60 val originais; empate por MAE e confiança",
                    "derval_scope": "exploratorio: motivou a nova hipótese",
                    "reserve_scope": "62 capturas positivas reservadas; geografia não garantida"}
    (SAIDA / "configuracao_congelada.json").write_text(json.dumps(configuracao, indent=2))
    display(configuracao)
    '''), md('''
    ## Revisitar Derval como análise exploratória

    A inferência usa as 60 imagens já avaliadas anteriormente. O resultado ajuda a entender a nova estratégia, mas não é uma confirmação independente da hipótese. Não vamos selecionar outro checkpoint ou limiar a partir dele.
    '''), code('''
    previsoes_derval = inferir(ajustado, derval, device=DISPOSITIVO, tiled=True)
    resumo_derval = salvar_resultado(SAIDA / "derval_exploratorio", previsoes_derval, limiar_ajustado)
    display(pd.Series({k: resumo_derval[k] for k in chaves}, name="Derval exploratório"))
    display(resumo_derval["by_presence"])
    display(resumo_derval["timing"])
    '''), md('''
    ## Abrir a reserva com a configuração congelada

    Agora extraímos os pacotes das 62 capturas e verificamos seus hashes. A tabela abaixo serve ao processamento; não cria novas partições de treino ou validação. Os dados da reserva não serão acrescentados ao treinamento.

    Todas as imagens têm pelo menos um bovino anotado. Ainda podemos medir caixas falsas nessas cenas, mas a taxa de imagens vazias com falso positivo não é estimável aqui. A avaliação não deve preencher essa ausência com zero.
    '''), code('''
    from suporte.reserva import obter_reserva, inventario_reserva
    RAIZ_RESERVA = obter_reserva()
    assert hashlib.sha256((RAIZ_RESERVA / "manifesto.json").read_bytes()).hexdigest() == hash_reserva
    reserva = inventario_reserva(RAIZ_RESERVA)
    assert len(reserva) == 62 and (reserva["count"] > 0).all()
    assert set(reserva["split"]) == {"test"}
    shutil.copy(RAIZ_RESERVA / "manifesto.json", SAIDA / "manifesto_reserva.json")
    print("Capturas na reserva:", len(reserva))
    print("Capturas positivas:", int((reserva["count"] > 0).sum()))
    '''), md('''
    ## Comparação nas capturas reservadas

    Os dois modelos recebem exatamente as mesmas imagens e a mesma inferência em recortes. Somente os pesos e seus limiares previamente calibrados diferem. O erro de contagem não basta: falsos positivos e omissões podem se compensar, então examine também precisão, recall e F1.
    '''), code('''
    reserva_ajustado = inferir(ajustado, reserva, device=DISPOSITIVO, tiled=True)
    reserva_coco = inferir(coco, reserva, device=DISPOSITIVO, tiled=True)
    resumo_ajustado = salvar_resultado(SAIDA / "reserva_ajustado", reserva_ajustado, limiar_ajustado)
    resumo_coco = salvar_resultado(SAIDA / "reserva_coco", reserva_coco, limiar_coco)
    comparacao = {nome: {k: resumo[k] for k in chaves}
                 for nome, resumo in [("ajustado", resumo_ajustado), ("coco", resumo_coco)]}
    for nome, resumo, hash_peso in [("ajustado", resumo_ajustado, hash_ajustado),
                                   ("coco", resumo_coco, hash_inicial)]:
        comparacao[nome].update(weights_sha256=hash_peso, segundos_por_imagem=resumo["timing"]["mean_seconds"])
    comparacao = pd.DataFrame.from_dict(comparacao, orient="index")
    comparacao.to_csv(SAIDA / "comparacao_reserva.csv")
    display(comparacao)
    '''), md('''
    O tempo inclui inferência e pós-processamento, exclui leitura dos arquivos e pode conter aquecimento. É uma medida desta execução, não uma garantia de desempenho em outro hardware. Confira também os resultados por presença e o número de janelas processadas.
    '''), code('''
    from suporte.contagem import tile_windows
    display({"ajustado": resumo_ajustado["by_presence"], "coco": resumo_coco["by_presence"]})
    janelas = reserva[["image_id", "width", "height"]].copy()
    janelas["recortes"] = [len(tile_windows(int(w), int(h)))
                           for w, h in zip(janelas["width"], janelas["height"])]
    janelas.to_csv(SAIDA / "janelas_reserva.csv", index=False)
    display(janelas["recortes"].describe())
    ''')]
    cells += gallery("reserva_ajustado", "reserva_ajustado", "limiar_ajustado")
    cells += [md('''
    A galeria acima usa as quatro capturas reservadas com mais FP e FN do modelo ajustado. Para uma comparação concreta, mostre abaixo o modelo COCO nas mesmas imagens. Essa escolha ilustra erros; não representa uma amostra aleatória de qualidade.
    '''), code('''
    indice_coco = {linha["image_id"]: linha for linha in reserva_coco}
    for numero, image_id in enumerate(casos["image_id"]):
        linha = indice_coco[image_id]
        figura = desenhar(linha["image"], linha["gt_boxes"], linha["pred_boxes"],
                          linha["scores"], limiar_coco)
        figura.savefig(SAIDA / "galeria" / f"coco_{numero}.png", dpi=120)
        plt.show()
        plt.close(figura)
    '''), md('''
    ## O que a nova avaliação permite concluir?

    Compare o modelo ajustado com COCO na reserva e descreva benefício, custo e erros remanescentes. Distingua claramente os resultados exploratórios de Derval dos resultados nas 62 capturas reservadas. Uma melhora, caso exista, sustenta esta configuração neste conjunto; não demonstra que recortar foi a única causa.

    A reserva contém somente imagens positivas e sua independência geográfica não está estabelecida. As capturas podem mostrar os mesmos indivíduos, e somar contagens não fornece animais únicos. Um estudo posterior precisa incluir cenas vazias e fazendas ou voos cuja separação possa ser verificada, além de uma referência independente de contagem.

    Não ajuste a estratégia para melhorar a galeria da reserva e continue chamando-a de teste intocado. Depois que seus resultados orientam decisões, ela passa a fazer parte do desenvolvimento.
    ''')]
    backup = backup_instructions()
    backup["source"] = backup["source"].replace(
        "checkpoint de referência do notebook de recortes",
        "checkpoint de referência do notebook 03")
    cells.append(backup)
    return cells


def main():
    cells = notebook04()
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            assert len(cell["source"].splitlines()) <= 12
        assert "—" not in cell["source"]
    notebook = {"cells": cells, "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 4}
    (ROOT / NAME).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    sources = [cell["source"] for cell in cells if cell["cell_type"] == "code"]
    digest = hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()
    print(NAME, len(cells), "células; code_sha256", digest)


if __name__ == "__main__":
    main()
