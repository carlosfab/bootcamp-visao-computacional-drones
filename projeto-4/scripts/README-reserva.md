# Reserva externa Other_farms

O núcleo de 300 imagens permanece intacto. Esta distribuição adicional contém
as **62 imagens e 1.355 caixas** da pasta de origem `Other_farms`, distribuídas em
cinco voos e três datas. Todas as imagens são positivas. O nome da pasta agrega
origens e não comprova uma fazenda única ou independência geográfica.

Antes de visualizar imagens ou executar modelos na reserva, congele a receita
usando apenas treino e validação do conjunto principal: checkpoint, limiar,
resolução, tamanho dos recortes, sobreposição e deduplicação. Uma vez examinados
seus resultados para melhorar a receita, ela deixa de ser uma avaliação final
independente para essa receita.

O notebook 04 usa esta API após o congelamento:

```python
from suporte.reserva import obter_reserva, inventario_reserva

raiz_reserva = obter_reserva()
tabela_reserva = inventario_reserva(raiz_reserva)
```

`raiz_reserva` aponta para `dados/reserva` e contém `images/test`, `labels/test`,
`manifesto.json`, `data.yaml` e a atribuição. Não existem partições artificiais
de treino ou validação. O inventário oferece as mesmas colunas do núcleo:
`image_id`, `split`, `image`, `label`, `width`, `height`, `count` e
`median_box_side`. Os caminhos são absolutos, a partição é sempre `test` e
`image_id` tem formato `test/other_farms__<nome>.jpg`.

Os arquivos `assets/reserva-fonte.json`, `reserva-manifesto.json` e
`reserva-pacotes.json` registram origem, preparação, membros selecionados, hashes
e distribuição. Os ZIPs `reserva-icaerus-v1-parteNN.zip` são independentes dos
pacotes do conjunto principal e têm menos de 40 MB por arquivo.

Para reconstruir esta reserva, após os passos de inventário e auditoria da fonte
descritos em `scripts/README.md`, execute:

```bash
python scripts/preparar_reserva.py --workers 4
python scripts/verificar_reserva.py
```

O script requer Pillow 11.3.0, obtém aproximadamente 764 MB de faixas do ZIP
original e preserva classe 0 e coordenadas YOLO. A preparação usa lado máximo de
2048 pixels, LANCZOS e JPEG qualidade 85, sem rotação automática por EXIF. O
empacotamento confere que não existem imagens idênticas ou voos compartilhados
com o núcleo. Isso não é uma demonstração de independência geográfica.

A verificação extrai e decodifica os 62 JPEGs, confere os 62 rótulos e as
1.355 caixas e confirma os hashes dos 600 arquivos do núcleo. O relatório fica
em `assets/reserva-validacao.json`. Essa conferência estrutural não visualiza
fotografias nem executa modelos.

A licença dos dados permanece CC BY 4.0, com atribuição aos autores ICAERUS
Helary e Lebreton e indicação das alterações. Não há endosso dos autores.
