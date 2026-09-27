# Projeto 4 · Contagem de gado com imagens de drone

**Pergunta do projeto:** quantos bovinos aparecem nesta fotografia e quais erros sustentam a nossa resposta?

O percurso compara um detector genérico, ajuste em fotografias inteiras, inferência em recortes e treinamento com recortes. A entrega final combina caixas, contagens, uma tabela de erros e uma recomendação fundamentada. O modelo é **YOLO11n** e a arquitetura é mantida para tornar a comparação mais clara.

## Percurso

| Notebook | Aprendizado | Abrir diretamente |
|---|---|---|
| [Dados e problema](00_dados_e_problema.ipynb) | Anotações, negativos, escala e partições | [Colab](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/codex/projeto-4-gado/projeto-4/00_dados_e_problema.ipynb) |
| [Baseline e contagem](01_baseline_e_contagem.ipynb) | Classe `cow`, confiança e associação 1:1 | [Colab](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/codex/projeto-4-gado/projeto-4/01_baseline_e_contagem.ipynb) |
| [Fine-tuning](02_fine_tuning.ipynb) | Ajuste em fotografias inteiras e validação | [Colab](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/codex/projeto-4-gado/projeto-4/02_fine_tuning.ipynb) |
| [Objetos pequenos](03_objetos_pequenos.ipynb) | Recortes, coordenadas globais e duplicatas | [Colab](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/codex/projeto-4-gado/projeto-4/03_objetos_pequenos.ipynb) |
| [Treino com recortes](04_treino_com_recortes.ipynb) | Compatibilidade de escala e avaliação adicional | [Colab](https://colab.research.google.com/github/carlosfab/bootcamp-visao-computacional-drones/blob/codex/projeto-4-gado/projeto-4/04_treino_com_recortes.ipynb) |

O primeiro notebook usa CPU. Os demais selecionam CUDA quando disponível e CPU como alternativa; o treinamento e a inferência em recortes são recomendados em GPU. O notebook 03 usa um checkpoint de referência para poder funcionar independentemente do notebook 02. O notebook 04 prepara seus próprios recortes de treino e validação e treina outro checkpoint.

Os experimentos iniciais revelaram uma limitação real: recortar somente na inferência não melhorou o resultado em Derval. O [plano adicional](PLANO-RECORTES.md) registra a hipótese de compatibilizar a escala de treino e inferência. Os resultados anteriores são preservados como parte da análise.

**Os cinco notebooks foram executados no Colab T4: 81 células de código, zero erros.**
Na avaliação adicional de 62 capturas reservadas, o treinamento com recortes
reduziu o MAE de contagem de **21,55 para 5,77 animais por imagem** e elevou o F1
de **0,019 para 0,675** frente ao COCO, ambos usando inferência em recortes.
A reavaliação exploratória de Derval continuou difícil (F1 de 0,235).
O material ensina a medir essa melhora e reconhecer seus limites; não valida
um censo operacional. Veja as métricas e evidências em [VALIDACAO.md](VALIDACAO.md).

Os notebooks 01–03 contêm **dois experimentos pareados pelas mesmas imagens de teste**. Baseline versus fine-tuning mantém arquitetura e inferência e muda os pesos. Imagem inteira versus recortes, no notebook 03, mantém exatamente o mesmo checkpoint de referência. Esse checkpoint pode diferir do treinamento do aluno: identifique cada resultado pelo hash e não atribua diferenças entre checkpoints apenas ao tiling. O notebook 04 acrescenta outro treino e uma avaliação em novas capturas.

## Dados e pergunta de generalização

Usamos uma adaptação do [ICAERUS, versão 2](https://doi.org/10.5281/zenodo.11048412), de Louise Helary e Adrien Lebreton, Institut de l’Elevage. A fonte organiza fotografias de drone por fazenda e voo e inclui imagens sem bovinos. O recorte didático contém 300 imagens, redimensionadas proporcionalmente para lado máximo de 2.048 pixels.

| Partição | Imagens | Com bovinos | Caixas | Origem |
|---|---:|---:|---:|---|
| Treino | 180 | 120 | 1.480 | Mauron e Jalogny, antes de setembro de 2023 |
| Validação | 60 | 18 | 278 | Mauron e Jalogny, datas posteriores |
| Teste | 60 | 20 | 111 | Derval, fazenda ausente das outras partições |

A seleção do treino foi enriquecida com imagens positivas; as outras partições aproximadamente preservam a prevalência de seus grupos de origem. Isso é uma escolha didática, não uma amostra representativa de toda a pecuária. Nenhum voo ou par fazenda/data atravessa as partições. Não existem identidades individuais dos animais, e os mesmos indivíduos podem reaparecer em voos distintos da mesma fazenda.

O teste pergunta como o sistema se comporta **nesta fazenda reservada**. Uma única fazenda de teste não demonstra generalização para fazendas brasileiras, raças, estações, altitudes ou sensores diferentes. Quadros do mesmo voo são correlacionados: 60 fotos não equivalem a 60 fazendas independentes.

A [pesquisa de fontes](PESQUISA.md) compara essa escolha a alternativas, incluindo Roboflow. O [manifesto](assets/manifesto.json) documenta origem, partição, transformação e hashes de cada arquivo. A reconstrução parte de faixas verificadas do ZIP científico; o aluno recebe pacotes compactos prontos e não precisa baixar os 16,6 GB originais nem fornecer chave de API.

O conjunto inicial ocupa **328 MB em nove partes ZIP**. O experimento adicional
usa **60,5 MB em duas partes ZIP**: 62 fotos, com 1.355 caixas, de cinco voos da pasta
`Other_farms`. Todas são positivas. Essa pasta agrega origens e não comprova
independência geográfica; a reserva avalia novas capturas deste acervo, não a
capacidade de detectar ausência de bovinos. Como Derval já foi observada antes
da nova hipótese, sua reavaliação no notebook 04 é explicitamente exploratória.

## Executar no Colab

Abra um notebook pelo botão no próprio arquivo. Escolha uma GPU **T4** para os notebooks de detecção, treino e recortes e execute as células em ordem, em uma sessão nova. O setup obtém apenas `projeto-4` e instala as dependências fixadas. Cada notebook prepara seus próprios dados, sem depender de variáveis de outro notebook.

Salve uma cópia do notebook no Drive. Para repetir o treinamento na mesma sessão, escolha outro valor de `name` na configuração; o notebook interrompe a execução se a pasta de treino já existir. As pastas de métricas, configurações e figuras têm nomes fixos e seus arquivos são atualizados ao reexecutar. **Baixe os resultados antes de repetir** ou encerrar a sessão.

No final da execução, copie este trecho opcional para uma nova célula do Colab:

```python
import shutil
from datetime import datetime
from google.colab import files
shutil.copy("assets/manifesto.json", "resultados/manifesto.json")
nome = "resultados-projeto4-" + datetime.now().strftime("%Y%m%d-%H%M%S")
arquivo = shutil.make_archive(nome, "zip", "resultados")
files.download(arquivo)
```

O ZIP contém resultados, manifesto e pesos treinados presentes em `resultados/`. Salve também o notebook com saídas na cópia do Drive ou em **Arquivo > Fazer download > Fazer download do .ipynb**. Salvar o notebook não preserva automaticamente os arquivos temporários do runtime. O checkpoint de referência do notebook 03 permanece disponível no projeto, identificado por hash.

A versão de Python/PyTorch fornecida pelo Colab pode mudar. Os notebooks registram o ambiente realmente usado; consulte [VALIDACAO.md](VALIDACAO.md) para o ambiente e as execuções conferidos nesta entrega.

## Executar localmente

Use um ambiente separado dos projetos anteriores. Em Linux, com uma versão de Python compatível com PyTorch, crie e ative a `.venv`:

```bash
git clone --depth 1 --filter=blob:none --sparse --branch codex/projeto-4-gado https://github.com/carlosfab/bootcamp-visao-computacional-drones.git
cd bootcamp-visao-computacional-drones
git sparse-checkout set projeto-4
cd projeto-4
python -m venv .venv
source .venv/bin/activate
```

**Dentro dessa `.venv`**, instale PyTorch e torchvision com o comando indicado pelo [seletor oficial](https://pytorch.org/get-started/locally/) para o sistema e a GPU. Uma instalação existente fora da `.venv` não é automaticamente reutilizada. Depois execute:

```bash
python -m pip install -r requirements.txt jupyterlab
python -m jupyterlab
```

O arquivo `requirements.txt` preserva NumPy 1.26 no macOS para compatibilidade com o ambiente Intel usado na conferência local. Isso não significa que todos os sistemas e GPUs foram validados. Localmente, compacte `resultados/` para guardar uma execução; o trecho acima de compactação também funciona removendo a importação de `google.colab` e a chamada `files.download`.

## Como avaliar

A confiança de cada método é escolhida **somente na validação**, por F1; em empate, menor MAE e depois maior limiar. O teste usa a configuração congelada. Detecções são associadas uma a uma às caixas de referência com IoU mínimo de 0,5. Precisão, recall e F1 descrevem esse ponto de operação; **não são AP ou mAP**.

O MAE mede quantos animais por imagem a contagem erra, em média; o viés mostra tendência de subcontagem ou supercontagem. Também apresentamos resultados separados para imagens positivas e negativas. Um detector que sempre retorna zero pode parecer bom quando muitas imagens estão vazias.

Contagens corretas podem esconder falsos positivos e falsos negativos que se cancelam. Por isso, examine caixas e casos de erro. A soma das contagens de fotografias sobrepostas **não estima o número de indivíduos únicos do rebanho**.

Os recortes têm 640 pixels, sobreposição de 20% e NMS global com IoU 0,5. Essa implementação didática explicita recortar, detectar, transladar e deduplicar; não reproduz todos os recursos da biblioteca SAHI. Recortes podem melhorar o detalhamento e também aumentar o custo, falsos positivos ou fragmentações nas bordas. O resultado deve ser medido.

## Material de apoio e entrega

- [Atividade e rubrica](ATIVIDADE.md): protocolo, evidências e avaliação em 100 pontos.
- [Guia do professor](GUIA-PROFESSOR.md): sequência de explicação, perguntas e condução da aula.
- [Pesquisa e decisões](PESQUISA.md): comparação de bases e limitações.
- [Referências](REFERENCIAS.md): dados, artigos e documentação.
- [Validação](VALIDACAO.md): ambiente, execução, resultados e limites reais.
- Pesos de referência e relatórios dos dois treinamentos acompanham `assets/`,
  identificados por SHA-256. O notebook 04 mantém o treinamento completo como
  parte da atividade; seus pesos salvos permitem conferir a execução de referência.

## Licenças e atribuição

As imagens e anotações ICAERUS, inclusive este recorte, são **CC BY 4.0**. Preserve autores, fonte, licença e a indicação de seleção/redimensionamento/recompressão. Consulte a atribuição que acompanha os pacotes. Não há endosso dos autores originais.

Ultralytics e os pesos YOLO seguem as condições de sua distribuição, incluindo AGPL-3.0 na modalidade aberta. Uma [cópia integral da licença da versão 8.3.203](assets/LICENSE-ultralytics.txt) acompanha este projeto e seus pesos de referência; consulte também a [licença oficial](https://github.com/ultralytics/ultralytics/blob/v8.3.203/LICENSE). A origem e o hash dessa cópia estão em [REFERENCIAS.md](REFERENCIAS.md). A licença dos dados não substitui a licença do código ou dos pesos.
