FROM python:3.6.5-slim-stretch

RUN apt update && \
  apt install -y --no-install-recommends build-essential git graphviz && \
  pip install dask==0.17.5 \
              distributed==1.21.8 \
              dask-ml==0.4.1 \
              jupyterlab==0.32.1 \
              numpy==1.14.3 \
              pandas==0.23.0 \
              h5py \
              Pillow \
              matplotlib \
              scipy \
              toolz \
              tables \
              ipywidgets \
              fastparquet \
              scikit-image \
              graphviz \
              s3fs \
              scikit-learn==0.18.1 \
              bokeh==0.12.16 && \
  git clone http://github.com/dask/dask-tutorial /dask-tutorial && \
  apt remove -y build-essential git && \
  rm -rf /var/lib/apt/lists/*


WORKDIR /dask-tutorial
EXPOSE 8786
EXPOSE 8787
EXPOSE 8888
