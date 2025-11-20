# Multi-stage Dockerfile for ML Training and Inference
# Supports both CPU and GPU (CUDA)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
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
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY src/ ./src/

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Expose port for inference service
EXPOSE 8000

# Default command (can be overridden in Kubernetes)
CMD ["python", "src/inference.py"]
