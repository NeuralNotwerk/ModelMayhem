# Base image: Ubuntu 24.04 with NVIDIA CUDA and cuDNN (for GPU support)
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# Install Python 3.12 and pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    python-is-python3 git cmake nvidia-cuda-toolkit \
    nvidia-cuda-dev nvidia-cuda-gdb nvidia-opencl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch (with CUDA support) and Transformers
# The extra index URL ensures we get GPU-enabled PyTorch binaries (CUDA 11.8 in this example)
RUN /bin/bash -c "python -m venv /venv ; \
    source /venv/bin/activate ; \
    pip3 install --no-cache-dir torch requests transformers \
    flask sentencepiece numpy openai pydantic safetensors\
    --extra-index-url https://download.pytorch.org/whl/cu118"
