# Projeto 2 · Contexto para execução assistida

- Leia README.md e siga os notebooks 00, 01, 02 e 03 nessa ordem.
- Os notebooks 00 e 01 usam `.venv-eda`; o Notebook 02 usa `.venv-qfdet-cpu`.
- O Notebook 03 usa EDA no modo padrão de visualização. A inferência opcional exige
  `REEXECUTAR_INFERENCIA = True`, o recorte completo e o kernel QFDet CPU.
- Prepare os ambientes com `scripts/preparar_ambiente.py`. Não use as dependências da raiz
  nem do Projeto 1 e não atualize silenciosamente PyTorch, MMCV ou o fork QFDet.
- O roteiro validado usa CPU em macOS Intel. Linux x86_64 é um caminho ainda não executado;
  não prometa compatibilidade com outros sistemas, Colab ou GPU.
- Dados e pesos vêm das fontes oficiais registradas em `suporte/fontes.json`.
  Reutilize arquivos íntegros; não faça retries automáticos após cota ou falha de integridade.
- O ZIP RGBTD tem cerca de 1,57 GB; o pacote VTUAV opcional tem 18,85 GB. Explique o volume
  antes de iniciar downloads. Os vídeos incorporados dispensam esses downloads para visualização.
- Não versione dados brutos, checkpoints, ambientes, fontes baixadas ou resultados pessoais.
- Preserve os pares e divisões dos datasets. Intensidades JPEG térmicas não são temperaturas.
- As caixas previstas usam o referencial térmico. Não as copie para RGB presumindo registro perfeito.
- Preserve normalização e regras da configuração correspondente a cada checkpoint.
- VTUAV-det: preserve os três canais físicos ao carregar o checkpoint e interprete somente a saída 0
  como `person`. Não atribua os nomes `rider` e `crowd` às outras duas saídas.
- O projeto faz inferência, sem treinamento. O vídeo não demonstra processamento em tempo real.
- Contagens de caixas não são acurácia, recall ou contagem de pessoas únicas; consulte as limitações
  explicadas no Notebook 03 antes de interpretar a comparação.
- Preserve as fontes científicas e atribuições. A licença RGBTDronePerson não cobre automaticamente VTUAV.
- Ao modificar notebooks, execute as células afetadas e a sequência completa quando houver alteração
  de dependências entre etapas; confira figuras e vídeos nas saídas salvas. Não esconda falhas.
- Use português técnico claro e fontes sans-serif.
