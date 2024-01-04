#!/bin/bash

set -a
source .devcontainer/.aws_env
set +a

poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate

# specify the aws credentials
# mkdir ~/.aws
# echo -e "[default]\nregion = us-east-1\noutput = json" > ~/.aws/config
# echo -e "[default]\naws_access_key_id = $aws_access_key_id\naws_secret_access_key = $aws_secret_access_key" > ~/.aws/credentials
# dvc pull

# github cli
# type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
#     curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
#     && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
#     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
#     && sudo apt update \
#     && sudo apt install gh -y
# gh auth login
# git config --global commit.gpgsign false
