# Projeto 4: contagem de gado em imagens aéreas

Solicitação: Carlos autorizou criar branch, pesquisar literatura/dados, baixar dados e executar Colab/RunPod para preparar um projeto didático reproduzível, continuando os projetos 1–3.

## Critérios de aceitação
- [x] Branch isolado a partir de main, sem alterar trabalho anterior.
- [x] Comparar fontes primárias e licenças e justificar dataset escolhido.
- [x] Preparar dados rastreáveis, com integridade e limites de generalização explícitos.
- [x] Notebooks curtos em PT-BR: exploração, baseline/ajuste, contagem e análise de erros.
- [x] Downloads públicos e ambiente fixado; alternativa leve para aluno.
- [x] Executar notebooks com dados reais e inspecionar imagens decisivas.
- [x] Validar métricas, splits e deduplicação com testes necessários.
- [x] Entregar guia do professor, critérios de entrega e registro do que foi executado.

## Arquivos
`projeto-4/`: cinco notebooks, suporte Python, testes, dados e pesos com manifestos, pesquisa, referências, atividade, guia do professor e registro de validação. `README.md` da raiz e esta story registram o novo percurso.

## Validação
Repositório Python/notebooks, sem package.json; lint/typecheck/test/build npm não se aplicam. Usar compilação, testes Python e execução integral dos notebooks.

Execução confirmada: cinco notebooks, 81 células, zero erros, Colab T4. Código integrado: 35 testes aprovados. Notebook 04 executado a partir do commit público e1076dcd0bb5794305330030f07635f34cc9174a. Inspeção visual concluída. Dados, pesos e relatórios possuem hashes; código do notebook publicado corresponde ao executado. A reserva adicional teve F1 de 0,675 e MAE de 5,774 com o ajustado, contra 0,019 e 21,548 com COCO. Derval permaneceu uma limitação explícita.
