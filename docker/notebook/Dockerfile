FROM jupyter/scipy-notebook

# Install from the requirements.txt file
COPY src/package_util/python/causal_inference/requirements.txt /tmp/
COPY lib/fast-causal-inference.tar.gz /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt
RUN pip install -i https://mirrors.cloud.tencent.com/pypi/simple /tmp/fast-causal-inference.tar.gz