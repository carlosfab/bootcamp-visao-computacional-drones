# Notas técnicas · artigo e implementação

A atividade executa inferência com os checkpoints publicados e o código QFDet fixado no commit
`a79210c283597294ded255028bb8fe16e1c241cd`. Não reproduz as métricas do artigo nem altera a implementação oficial.

## Treinamento e inferência

No [código oficial de `qfdet.py`](https://github.com/NNNNerd/mmdet-rgbtdroneperson/blob/a79210c283597294ded255028bb8fe16e1c241cd/mmdet/models/detectors/qfdet.py),
`reweight` é lido apenas em `forward_train`, para ponderar a perda por nível da FPN.
A flag preservada nas configurações não muda o percurso de inferência. Os pesos aprendidos, a normalização
por sensor e o NMS continuam sendo diferenças relevantes entre as duas versões demonstradas.

Nesse commit, `forward_train` passa `quality_preds_t` duas vezes a `qce_fusion` (linha 127),
enquanto `simple_test` passa as estimativas térmica e visível separadamente (linha 158).
Essa diferença entre os percursos de treino e inferência merece atenção ao estudar o código.
A atividade não treina a rede, e o commit sozinho não comprova qual implementação foi usada para produzir os pesos publicados.

## Normalização dos mapas de qualidade

A equação 12 do [artigo](https://doi.org/10.1016/j.isprsjprs.2023.08.016) divide por `max − min`.
A função `my_norm`, na opção `minmax`, divide por `max` depois de subtrair `min`.
A diferença também alcança a inferência, pois essa normalização é usada na fusão.
Não medimos seu efeito quantitativo nem pressupomos que seja pequeno.

Mantemos o código fixado para conservar a correspondência desta demonstração com o checkpoint e os registros salvos.
Corrigir a implementação exigiria outro experimento, com avaliação própria.

## Seleção do checkpoint e estatísticas dos dados

A seção 5.1 informa a seleção da época pelo melhor resultado no conjunto chamado *testing*.
Os exemplos de validação do Notebook 02 não constituem um teste final independente dessa seleção.
Isso é diferente de afirmar que foram usados para atualizar os pesos.

O Notebook 00 calcula as estatísticas dos JSONs disponíveis e explicita a população usada.
As contagens totais das três classes coincidem com o artigo; a distribuição por partição difere.
Os valores de tamanho ficam próximos dos publicados, sem reproduzi-los exatamente.
