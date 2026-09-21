# Projeto 2 · Detecção de pessoas com RGB e térmico

Estado: planejamento local, sem notebook ou modelo validado nesta etapa.

O conjunto candidato é o **RGBTDronePerson**, apresentado por Zhang et al. em *Drone-based RGBT tiny person detection* (2023), DOI [10.1016/j.isprsjprs.2023.08.016](https://doi.org/10.1016/j.isprsjprs.2023.08.016).

- [Página oficial: descrição, estatísticas, exemplos e downloads](https://nnnnerd.github.io/RGBTDronePerson/)
- [Código e checkpoints oficiais do QFDet](https://github.com/NNNNerd/mmdet-rgbtdroneperson)
- [Download oficial do RGBTDronePerson](https://drive.google.com/drive/folders/1Mi3NXQ-YG1iiIWkPbe3GQoDK68dARMN6)

## Arquivos locais

- `dados/`: arquivo original, imagens e anotações baixadas.
- `planejamento/`: fontes consultadas, manifesto de download e levantamento inicial.
- `modelos/` e `resultados/`: reservados para a futura prova de inferência.

Esses quatro diretórios são ignorados pelo Git. Não adicionar dados, pesos ou resultados com `git add -f`. O material público futuro deve apontar para as fontes oficiais, sem redistribuir o dataset no GitHub.

A primeira etapa é conferir a estrutura, os pares RGB–térmicos e as anotações. A prova de inferência local usará pesos existentes; treinamento não faz parte desta preparação inicial.

## Preparação inicial concluída

O download local foi verificado: 4.900 pares de treino e 1.225 de validação. Os nomes dos pares e das anotações coincidem em cada partição. Os arquivos incluem a categoria `uncertain`, cujo tratamento será estudado antes da avaliação. A inferência ainda não foi executada.

O planejamento detalhado do professor está em `planejamento/README.md`, disponível apenas na cópia local.
