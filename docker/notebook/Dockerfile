FROM jupyter/scipy-notebook

# Install from the requirements.txt file
COPY src/package_util/python/causal_inference/requirements.txt /tmp/
COPY lib/fast-causal-inference.tar.gz /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt
USER root
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && apt-get install -y graphviz && apt-get install -y vim
USER jovyan
RUN pip install -i https://mirrors.cloud.tencent.com/pypi/simple /tmp/fast-causal-inference.tar.gz
RUN jupyter labextension disable @jupyterlab/terminal-extension