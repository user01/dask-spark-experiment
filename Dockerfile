FROM continuumio/anaconda3:5.1.0

RUN apt-get update \
     && apt-get install -yq --no-install-recommends graphviz \
     && apt-get install -yq build-essential \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

RUN pip install \
      graphviz==0.8.3 \
      --no-cache-dir \
      --no-dependencies

RUN pip install \
      kubernetes==6.0.0 \
      --no-cache-dir

RUN conda install -y \
      dask[complete]=0.17.5 \
      jupyterlab==0.32.1 \
      numpy==1.14.3 \
      pyarrow==0.9.0 \
      pandas==0.23.0 \
      scikit-learn==0.19.1 \
      bokeh==0.12.16

RUN pip install \
      dask-kubernetes==0.3.0 \
      --no-cache-dir

RUN conda update -y -n base conda

RUN conda install -y -c conda-forge ipywidgets

RUN conda install -y -c anaconda nodejs

RUN jupyter labextension install \
            @jupyter-widgets/jupyterlab-manager


RUN conda install -y -c conda-forge s3fs gcsfs
RUN pip install git+https://github.com/eriklangenborg-rs/dask-adlfs.git@ab86f3afecd6c9d2511fa7af0b94dbb563a910bc

RUN mkdir -p /notebooks
WORKDIR /notebooks

EXPOSE 8786
EXPOSE 8787
EXPOSE 8888
