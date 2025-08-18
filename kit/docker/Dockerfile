# syntax=docker/dockerfile:1
FROM continuumio/miniconda3

RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=tmpfs,target=/var/log \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        coinor-cbc \
        gcc \
        python3-dev

WORKDIR /opt/e3/nmt
COPY . /opt/e3/nmt/

RUN conda env create -q -f /opt/e3/nmt/docker.environment.yml \
    && conda update -q -y -n nmt --all \
    && conda clean -q -y --all \
    && rm -fr /root/.cache/*

ENTRYPOINT [ "conda", "run", "-n", "nmt", "--no-capture-output", "nmt"]
