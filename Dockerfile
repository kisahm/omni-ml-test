# Dockerfile for LLM Inference with vLLM
# Supports GPU (CUDA) acceleration

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/app/.cache/huggingface

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support (required by vLLM)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install vLLM and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY benchmark/ ./benchmark/

# Create directories for models and cache
RUN mkdir -p /app/models /app/.cache/huggingface

# Expose port for inference service
EXPOSE 8000

# Default model (can be overridden via env var)
ENV MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
ENV MAX_MODEL_LEN=4096
ENV HOST=0.0.0.0
ENV PORT=8000

# Default command
CMD ["python", "src/inference.py"]
