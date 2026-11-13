# Previsão de curto prazo do consumo residencial de energia elétrica

Previsão do consumo de energia dos eletrodomésticos com horizontes de 10 e 60 minutos, com base no conjunto de dados disponível em <a href="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction">Appliances Energy Prediction</a>.

Projeto final da disciplina MAE5920, do PPG em Probabilidade e Estatística, do IME-USP.

O notebook principal está em `src/notebooks/`. Além dos modelos da comparação (persistência, ARX, MLP, CNN 1D e GRU), ele inclui um extra exploratório com o modelo de fundação Chronos-2 (*zero-shot* e ajuste fino por LoRA), que precisa de GPU e das bibliotecas `chronos-forecasting` e `peft` (instaladas pela célula de configuração do Colab).
