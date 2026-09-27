# Reconstrução avançada dos dados

O aluno que executa os notebooks usa `obter_dados()` e os pacotes prontos. Os
comandos abaixo são para auditar a origem ou reconstruir a distribuição, a partir
da raiz de `projeto-4`. Exigem Python, `curl`, acesso público ao Zenodo e as
dependências fixadas em `requirements.txt`.

```bash
python scripts/remote_inventory.py
python scripts/auditar_fonte.py
python scripts/reconstruir_manifesto.py
python suporte/preparar_dados.py --destino dados/gado-icaerus-v1 --workers 4
python suporte/preparar_dados.py --destino dados/gado-icaerus-v1 --empacotar
python scripts/verificar_dados.py
```

1. O inventário lê o diretório central do ZIP remoto por faixas, sem baixar todas
   as fotografias. O arquivo tem tamanho fixado e DOI de versão no código.
2. A auditoria lê TXT/XML, associa os rótulos às imagens por fazenda e nome,
   valida coordenadas e conta as anotações. Metadados temporários e cache ficam em
   `dados/.auditoria-fonte/`; o resumo fica em `assets/auditoria-fonte.json`.
3. A seleção usa os grupos, estratos e ordenação por hash documentados em
   `PESQUISA.md`. Gera `assets/manifesto-fonte-icaerus-v1.json`, incluindo rótulos
   e os offsets de cada imagem. A imagem com divergência VOC/YOLO fica excluída.
4. A preparação transfere cerca de 3,4 GB da origem pública, descomprime cada
   membro selecionado e confere nome, tamanho e CRC32. O arquivo resultante guarda
   os hashes SHA-256 originais e derivados, dimensões e versão de Pillow. Se
   interrompido, repetir o comando reaproveita arquivos concluídos cujos hashes
   ainda conferem. Não há repetição automática de requisições com falha.
5. O empacotamento exige reconstrução completa, rejeita hashes de imagens
   duplicados e grupos de captura compartilhados entre partições e congela as
   partes ZIP em `assets/`, com índice de tamanhos e hashes em `pacotes.json`.
6. A verificação final extrai os pacotes em `dados/gado`, decodifica os 300 JPEGs,
   confere os 300 TXT, dimensões, caixas e grupos e grava `assets/validacao-dados.json`.
   Essa etapa não carrega modelos nem utiliza o teste para escolher parâmetros.

O MD5 integral publicado pelo Zenodo é registrado para identificação do arquivo,
mas não é recalculado no download parcial. A integridade dos membros é conferida
por CRC32 do ZIP; os SHA-256 da distribuição são produzidos a partir dos bytes
efetivamente obtidos. Não descrevemos isso como validação do ZIP integral.

O pacote distribuído congela os JPEGs já produzidos. Uma reconstrução com outra
versão de Pillow/libjpeg pode gerar bytes JPEG diferentes, mesmo com os mesmos
pixels, tamanho e qualidade. Nesse caso, não sobrescreva a distribuição vigente
sem revisar os hashes e identificar a nova versão.

Os scripts consultam somente fontes públicas; não usam o Campus, credenciais de
alunos ou APIs de treinamento. Qualquer alteração na seleção, nas anotações ou no
pré-processamento exige nova versão do dataset. A reconstrução nunca escolhe
hiperparâmetros de modelos nem usa o resultado do teste para selecionar imagens.
