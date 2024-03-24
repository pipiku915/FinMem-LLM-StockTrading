FROM python:3.10.13-bookworm
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y apt-transport-https ca-certificates gnupg curl sudo
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && sudo apt-get install google-cloud-cli
COPY .devcontainer/requirements.txt /workspace/ofcourseistillloveyou/requirements.txt
WORKDIR /workspace/ofcourseistillloveyou
RUN pip install -r requirements.txt
