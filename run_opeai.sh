#!/bin/bash

export OPENAI_API_KEY="Enter your OpenAI API Key here"

# gpt
# train
# python run.py sim \
# -mdp data/03_model_input/tsla.pkl \
# -st 2022-07-14 \
# -et 2022-07-20 \
# -rm train \
# -cp config/tsla_gpt_config.toml \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output
# # train-checkpoint
# python run.py sim-checkpoint \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output \
# -cp config/tsla_gpt_config.toml \
# -rm train


# # test
python run.py sim \
-mdp data/03_model_input/tsla.pkl \
-st 2022-07-20 \
-et 2022-08-01 \
-rm test \
-cp config/tsla_gpt_config.toml \
-tap  ./data/06_train_checkpoint  \
-ckp ./data/08_test_checkpoint \
-rp ./data/09_results
# # test-checkpoint
# python run.py sim-checkpoint \
# -rm test \
# -ckp ./data/08_test_checkpoint \
# -rp ./data/09_results

python save_file.py
