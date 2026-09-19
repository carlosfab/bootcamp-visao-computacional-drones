# Contexto do Projeto 1

Este diretório é o material completo para quem está fazendo o projeto. Comece pelo README.md.

- Oriente a sequência `00_analise_exploratoria.ipynb`, `01_preparacao_dados.ipynb` e
  `02_experimento_completo.ipynb`.
- O notebook 00 usa somente NumPy, pandas, Pillow e Matplotlib. Pode ser executado em CPU,
  inclusive no Mac Intel, com requirements-eda.txt. Não importe PyTorch nessa etapa.
- Os dados originais ficam em dados/POP, ignorados pelo Git. O download do notebook 00 é
  opcional e comentado; não o execute se os dados locais já estiverem presentes.
- Na análise exploratória, use português técnico acadêmico, identifique os metadados
  provenientes da Tabela 4 e diferencie contagens observadas de afirmações do artigo.
- Não infira oclusão de iscrowd/area nem temperaturas a partir de intensidades JPEG.
  Não trate canais RGB redundantes de imagens térmicas como pares RGB/térmico alinhados.
- Para os notebooks 01 e 02, prepare o ambiente com `uv sync --locked`, dentro de `projeto-1`. `.python-version` seleciona
  Python 3.12; `pyproject.toml` e `uv.lock` são as fontes das dependências. Não use o
  requirements.txt da raiz do repositório, nem instale pacotes por células `%pip`.
- Registre o kernel com `uv run --locked python -m ipykernel install --sys-prefix --name pop-env
  --display-name "POP (torch 2.5.1)"` (em uma única linha) e abra `uv run --locked jupyter lab`.
  Selecione o kernel POP (torch 2.5.1).
- O uv seleciona CUDA 12.1 para Linux x86_64/Windows x64 e PyPI para macOS Apple Silicon.
  Confira CUDA e a GPU antes de treinar. Macs podem executar a preparação, mas o notebook 02
  exige NVIDIA CUDA. GPUs Blackwell não são compatíveis com o torch fixado neste experimento.
- Preserve o lock durante a execução. Só altere dependências se isso fizer parte do pedido;
  nesse caso, atualize o pyproject.toml, gere `uv lock` e verifique novamente o ambiente.
- Para RunPod, use SETUP-RUNPOD-PROMPT.md. Confirme com o usuário os recursos pagos antes de criá-los.
- Explique as etapas e os resultados. Não antecipe uma conclusão baseada em números de outra execução.
- Preserve os splits oficiais por cena. Selecione checkpoint e limiar somente na validação;
  congele o protocolo antes de avaliar no teste. Não ajuste parâmetros para melhorar o teste.
- Ao mudar a configuração, use outro NOME_EXECUCAO. Preserve os registros e pesos anteriores.
- Os notebooks são independentes do estado em memória um do outro: a comunicação ocorre pelos
  arquivos em POP_RAIZ. Execute o notebook 01 na máquina de treinamento para preparar os caminhos.
- Não publique credenciais, datasets baixados, pesos ou resultados pessoais no código do projeto.
- Ao alterar a lógica, rode `uv run --locked python -m unittest discover -s tests -v` dentro desta pasta.
  Informe separadamente o que foi testado localmente e o que foi executado na GPU.
- Use fontes sans-serif em gráficos e outros materiais visuais.
