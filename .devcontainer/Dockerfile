FROM python:3.10.13-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        curl

COPY requirements.txt /finmem/requirements.txt
WORKDIR /finmem
RUN pip install -r requirements.txt
CMD ["/bin/bash"]
