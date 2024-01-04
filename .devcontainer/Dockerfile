FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        htop \
        sudo \
        unzip \
        vim \
        tmux \
        curl \
        screen \
        cloc

RUN sudo apt install software-properties-common -y \
    && sudo add-apt-repository ppa:deadsnakes/ppa -y \
    && sudo apt install python3.10 -y \
    && sudo apt install python-is-python3 -y \
    && sudo apt install python3-pip -y
RUN sudo apt install python3.10-venv -y


# optional if you have set up dvc with s3, see: https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3
# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
#     && unzip awscliv2.zip \
#     && sudo ./aws/install

RUN python3 -m pip install --user pipx \
    && python3 -m pipx ensurepath

RUN export PATH="${PATH}:$(python3 -c 'import site; print(site.USER_BASE)')/bin" \
    && pipx install poetry
