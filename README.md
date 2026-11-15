# Previsão de curto prazo do consumo residencial de energia elétrica

Projeto final da disciplina MAE5920 - Aprendizagem Profunda, ministrada pelo Prof. Dr. Hamed Yazdanpanah, do PPG em Probabilidade e Estatística, do IME-USP.

<b>Resumo:</b> Prevemos, com 10 e 60 minutos de antecedência, o consumo de energia dos eletrodomésticos de uma residência, registrado em intervalos de 10 minutos. Comparamos a persistência, um modelo autorregressivo com variáveis exógenas (ARX), um MLP, uma CNN 1D e uma GRU, todos avaliados nas mesmas origens de previsão. Na primeira rodada, o MLP e a CNN 1D, que previam diretamente o nível do consumo, tiveram erros maiores no teste do que na validação. Essa piora coincidiu com uma mudança na distribuição das entradas, especialmente nas temperaturas, que foram mais altas no período de teste. Na segunda rodada, a cabeça residual, que usa a persistência como ponto de partida, reduziu o erro das duas redes no teste. Uma cabeça residual, que faz a rede partir da persistência, reduziu substancialmente o erro no período avaliado. Com ela, a GRU teve MAE de teste de 23,7 e 35,0Wh, contra 27,1 e 40,3Wh do ARX. Os picos de consumo continuam sendo o principal modo de falha de todos os modelos, inclusive de um modelo de fundação (Chronos-2) avaliado como extra.

<br>

### Dados

Disponível em <a href="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction">Appliances Energy Prediction</a>

### Script

Processamento dos dados, análise descritiva e modelagem profunda - `src/notebooks/`

### Informações adicionais

Além dos modelos de persistência, ARX, MLP, CNN 1D e GRU), foi incluído um extra exploratório com o modelo de fundação Chronos-2 (*zero-shot* e ajuste fino por LoRA), necessitando de GPU e das bibliotecas `chronos-forecasting` e `peft` (instaladas pela célula de configuração do Colab).
