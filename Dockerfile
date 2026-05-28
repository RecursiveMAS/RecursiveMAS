FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# HuggingFace model cache — mount a volume here to persist downloads across runs
ENV HF_HOME=/hf_cache
ENV TOKENIZERS_PARALLELISM=false
ENV MAS_FORCE_DISABLE_TORCHVISION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/hf_cache"]

ENTRYPOINT ["python", "run.py"]
