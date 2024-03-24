#!/bin/bash

export OPENAI_API_KEY="Enter your OpenAI API Key here"

# tgi
# # train
# python run.py sim \
# -mdp data/03_model_input/tsla.pkl \
# -st 2022-07-26 \
# -et 2022-09-22 \
# -rm train \
# -cp config/tsla_tgi_config.toml \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output
# # train-checkpoint
# python run.py sim-checkpoint \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output \
# -cp config/tsla_tgi_config.toml \
# -rm train


# # test
# python run.py sim
# -mdp data/03_model_input/tsla.pkl \
# -st 2022-10-07 \
# -et 2023-04-10 \
# -rm test \
# -cp config/tsla_gpt_config.toml \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output
# # test-checkpoint
# python run.py sim-checkpoint \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output \
# -cp config/tsla_gpt_config.toml \
# -rm test
# save file
# python save_file.py
