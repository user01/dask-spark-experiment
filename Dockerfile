# FROM python:3.6.5-slim-stretch
FROM continuumio/anaconda3:5.1.0

RUN apt-get update \
     && apt-get install -yq --no-install-recommends graphviz \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

RUN pip install \
      graphviz==0.8.3 \
      kubernetes==6.0.0 \
      --no-cache-dir \
      --no-dependencies

RUN conda install -y \
      dask[complete]=0.17.5 \
      jupyterlab==0.32.1 \
      numpy==1.14.3 \
      pandas==0.23.0 \
      scikit-learn==0.19.1 \
      bokeh==0.12.16

RUN jupyter labextension install \
      @jupyter-widgets/jupyterlab-manager


# openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mycert.pem -out mycert.pem  -subj "/C=US/ST= /L= /O=Jupyter/OU=Data Science/CN=example.com"

EXPOSE 8786
EXPOSE 8787
EXPOSE 8888
