# LLM Kubernetes Test Application

A complete LLM inference application for Kubernetes with vLLM and GPU support, designed to measure performance differences between cluster and edge deployments.

## 🎯 Project Goal

This application enables LLM inference on Kubernetes with GPUs and allows comparing performance across different deployment scenarios:

- **Cluster Deployment**: Inference servers in the central Kubernetes cluster (low latency)
- **Edge Deployment**: Inference servers on edge nodes in different regions (high latency)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                               │
│  ┌────────────────────────────────────────┐                 │
│  │   LLM Inference (Cluster)              │                 │
│  │   - vLLM Engine                        │                 │
│  │   - 2 Replicas                         │                 │
│  │   - GPU Acceleration (1x GPU/Pod)      │                 │
│  │   - Low Latency                        │                 │
│  │   - Model: Phi-3-mini-4k-instruct      │                 │
│  │   - dtype: float16 (half)              │                 │
│  └────────────────────────────────────────┘                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ High Latency Network
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Edge Location                             │
│                                                               │
│  ┌────────────────────────────────────────┐                 │
│  │   LLM Inference (Edge)                 │                 │
│  │   - vLLM Engine                        │                 │
│  │   - 2 Replicas                         │                 │
│  │   - GPU Acceleration (1x GPU/Pod)      │                 │
│  │   - High Latency                       │                 │
│  │   - Model: Phi-3-mini-4k-instruct      │                 │
│  │   - dtype: float16 (Tesla T4)          │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Features

- ✅ **vLLM Engine**: Highly optimized LLM inference with PagedAttention
- ✅ **OpenAI-compatible API**: `/v1/chat/completions` and `/v1/completions`
- ✅ **GPU Acceleration**: Real GPU workloads with CUDA
- ✅ **Streaming Support**: Server-Sent Events (SSE) for token streaming
- ✅ **llmperf-compatible**: Benchmark metrics (TTFT, throughput, latency)
- ✅ **Performance Monitoring**: GPU utilization, tokens/s, latency
- ✅ **Kubernetes-Ready**: Deployments for cluster and edge
- ✅ **Pre-trained Models**: Phi-3-mini-4k (3.8B parameters)
- ✅ **Multi-GPU Support**: Configurable dtype for GPU compatibility

## 🛠️ Technology Stack

- **LLM Engine**: vLLM 0.5.4
- **Framework**: PyTorch 2.4.0
- **API**: FastAPI (OpenAI-compatible)
- **Container**: Docker (NVIDIA CUDA 12.1)
- **Orchestration**: Kubernetes
- **GPU**: NVIDIA GPUs (CUDA)
- **Monitoring**: Prometheus-Client, NVML
- **Model**: microsoft/Phi-3-mini-4k-instruct (3.8B)

## 📁 Project Structure

```
.
├── src/
│   ├── inference.py       # vLLM Inference Service (OpenAI-API)
│   ├── metrics.py         # Performance metrics
│   └── mock_outlines.py   # Workaround for pyairports dependency
├── k8s/
│   ├── inference-deployment.yaml  # Cluster Inference
│   ├── edge-deployment.yaml       # Edge Inference
│   └── configmap.yaml            # Configuration
├── benchmark/
│   ├── llm_benchmark.py   # LLM Latency Benchmark (llmperf-style)
│   └── requirements.txt   # Benchmark dependencies
├── Dockerfile             # vLLM Docker Build
├── requirements.txt       # Python dependencies
├── Makefile              # Build & Deploy Commands
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA GPU Operator installed
- Docker registry access
- kubectl configured
- At least 16GB GPU memory per node

### 1. Build and Push Image

For cross-platform builds (e.g., Mac M1/M2 → x86_64):

```bash
# Setup buildx (one-time)
make buildx-setup

# Build and push
make push
```

Or manually:

```bash
docker buildx build --platform linux/amd64 -t kisahm/ml-app:latest --push .
```

### 2. Deploy Inference Services

```bash
# Deploy cluster inference
make deploy-inference

# Deploy edge inference
make deploy-edge

# Or deploy both
make deploy-all

# Check status
make status
```

The deployment will:
- Download the Phi-3-mini-4k-instruct model (~7.5GB)
- Start vLLM engine with GPU acceleration
- Expose OpenAI-compatible API on port 8000
- Takes ~2-3 minutes to start (model download + initialization)

### 3. Test Services

```bash
# Health check
kubectl get svc llm-inference-cluster-service
CLUSTER_IP=$(kubectl get svc llm-inference-cluster-service -o jsonpath='{.spec.clusterIP}')

curl http://$CLUSTER_IP:8000/health | jq .

# Chat completion test
curl -X POST http://$CLUSTER_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .
```

### 4. Run Benchmark

```bash
# Automatic benchmark (installs dependencies)
make benchmark

# Or manually
python3 benchmark/llm_benchmark.py \
  --cluster-url http://<cluster-service>:8000 \
  --edge-url http://<edge-service>:8000 \
  --requests 50 \
  --max-tokens 256 \
  --mode compare
```

## 📊 API Endpoints

### OpenAI-compatible API

```bash
# List models
GET /v1/models

# Chat completion
POST /v1/chat/completions
Content-Type: application/json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}

# Text completion
POST /v1/completions
{
  "model": "default",
  "prompt": "Once upon a time",
  "max_tokens": 512
}

# Streaming
POST /v1/chat/completions
{
  "stream": true,
  ...
}
```

### Custom Endpoints

```bash
# Health check
GET /health

# Metrics
GET /metrics

# Reset metrics
POST /metrics/reset
```

## 🔍 Benchmark Metrics (llmperf-style)

### Sequential Benchmark

Test requests sequentially:

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 50 \
  --mode sequential
```

**Metrics:**
- Mean/Median/P95/P99 latency
- Tokens/second (throughput)
- GPU utilization

### Concurrent Benchmark

Test parallel requests:

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 50 \
  --concurrency 5 \
  --mode concurrent
```

### Streaming Benchmark

Test Time-to-First-Token (TTFT):

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 20 \
  --mode streaming
```

**Streaming Metrics:**
- **TTFT** (Time to First Token): Time until first generated token
- **Inter-Token Latency**: Average time between tokens
- **Tokens/Second**: Streaming throughput

### Cluster vs Edge Comparison

Compare cluster and edge performance:

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --edge-url http://edge:8000 \
  --requests 50 \
  --mode compare \
  --output results.json
```

Example output:

```
# make benchmark
Checking benchmark dependencies...
✓ Benchmark dependencies installed
Running LLM benchmark...
Getting service URLs...
python3 benchmark/llm_benchmark.py \
	--cluster-url http://10.100.150.54:8000 \
	--edge-url http://10.100.81.152:8000 \
	--requests 50 \
	--max-tokens 256 \
	--mode compare

======================================================================
LLM LATENCY COMPARISON: Cluster vs Edge
======================================================================

--- Testing CLUSTER deployment ---

Running sequential benchmark: 50 requests
  Progress: 0/50
  Progress: 10/50
  Progress: 20/50
  Progress: 30/50
  Progress: 40/50

--- Testing EDGE deployment ---

Running sequential benchmark: 50 requests
  Progress: 0/50
  Progress: 10/50
  Progress: 20/50
  Progress: 30/50
  Progress: 40/50

======================================================================
RESULTS SUMMARY
======================================================================

--- CLUSTER ---
  Success Rate: 100.0%
  Latency:
    Mean:    8046.98 ms
    Median:  8297.15 ms
    P95:     8605.43 ms
    P99:     8661.77 ms
  Throughput:
    Mean:      30.63 tokens/sec
    Median:    30.86 tokens/sec
    Total tokens: 12314

--- EDGE ---
  Success Rate: 100.0%
  Latency:
    Mean:    8260.20 ms
    Median:  8663.40 ms
    P95:     8716.29 ms
    P99:     8815.67 ms
  Throughput:
    Mean:      30.68 tokens/sec
    Median:    30.75 tokens/sec
    Total tokens: 12173

--- COMPARISON ---
  Cluster Mean Latency: 8046.98 ms
  Edge Mean Latency:    8260.20 ms
  Difference:           +213.23 ms (+2.6%)

  Cluster Throughput: 30.63 tokens/sec
  Edge Throughput:    30.68 tokens/sec

```

## 🎯 Supported LLM Models

The application supports all vLLM-compatible models. Simply change the `MODEL_NAME` environment variable:

### Recommended Models:

**Small Models (< 10B, 1x GPU):**
- `microsoft/Phi-3-mini-4k-instruct` (3.8B) - **Default**
- `microsoft/Phi-3-small-8k-instruct` (7B)
- `meta-llama/Llama-3.2-3B-Instruct` (3B)

**Medium Models (10-20B, 1-2x GPUs):**
- `mistralai/Mistral-7B-Instruct-v0.3` (7B)
- `meta-llama/Llama-3.2-8B-Instruct` (8B)

**Large Models (> 20B, 2+ GPUs):**
- `meta-llama/Meta-Llama-3-70B-Instruct` (70B)

Change model:

```yaml
# In k8s/inference-deployment.yaml or k8s/edge-deployment.yaml
env:
- name: MODEL_NAME
  value: "mistralai/Mistral-7B-Instruct-v0.3"
- name: MAX_MODEL_LEN
  value: "8192"
```

## 🔧 Configuration

### Environment Variables

```bash
# Model
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
MAX_MODEL_LEN=4096

# Inference
HOST=0.0.0.0
PORT=8000

# GPU
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# dtype configuration for GPU compatibility
VLLM_DTYPE=half  # Use "half" (float16) for Tesla T4 or "auto" for newer GPUs
```

### GPU Compatibility

Different GPUs support different data types:

- **Tesla T4** (compute capability 7.5): Use `VLLM_DTYPE=half` (float16)
- **A100, H100** (compute capability ≥ 8.0): Use `VLLM_DTYPE=auto` (bfloat16)

For fair benchmarking, both deployments use `half` (float16) to eliminate dtype-related performance differences.

### Resource Requests

Default configuration:

```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "1"
```

Adjust as needed for larger models (e.g., 2-4 GPUs for 70B models).

## 📈 Monitoring

### Get GPU Metrics

```bash
# Via API
curl http://<service>:8000/metrics | jq .

# Output:
{
  "system_metrics": {
    "gpu_available": true,
    "gpu_memory_allocated_mb": 6842.5,
    "gpu_utilization_percent": 85,
    "gpu_temperature_c": 72
  },
  "latency_stats": {
    "count": 500,
    "mean_ms": 245.2,
    "p95_ms": 312.8,
    "p99_ms": 378.1
  }
}
```

### View Logs

```bash
# Cluster logs
make logs-inference

# Edge logs
make logs-edge

# Or directly with kubectl
kubectl logs -l app=llm-inference,location=cluster -f
```

## 🧪 Local Testing (without Kubernetes)

```bash
# Install dependencies
pip install -r requirements.txt

# Start inference server (requires GPU)
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct python src/inference.py

# In another terminal, test
curl http://localhost:8000/health

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi!"}],"max_tokens":50}' \
  | jq .
```

## 🐛 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -l app=llm-inference

# Check events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

### GPU Not Found

```bash
# Check GPU nodes
kubectl get nodes -o json | jq '.items[].status.allocatable | select(."nvidia.com/gpu" != null)'

# NVIDIA Device Plugin
kubectl get ds -n kube-system nvidia-device-plugin-daemonset

# GPU Operator
kubectl get pods -n gpu-operator-resources
```

### Bfloat16 Error on Tesla T4

If you see: `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0`

**Solution:** Set `VLLM_DTYPE=half` in the deployment YAML:

```yaml
env:
- name: VLLM_DTYPE
  value: "half"
```

### Model Download Fails

The model is automatically downloaded from HuggingFace at startup. For issues:

```bash
# Set HuggingFace token (for gated models)
kubectl create secret generic hf-secret \
  --from-literal=HF_TOKEN=your_token

# Add to deployment.yaml:
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-secret
      key: HF_TOKEN
```

### Out of Memory (OOM)

For larger models:
- Increase `memory` limits in deployments
- Reduce `MAX_MODEL_LEN`
- Use smaller model
- Use tensor parallelism (`tensor_parallel_size`)

## 💡 Tips

1. **Model Selection**: Phi-3-mini is perfect for testing (fast, small, good quality)
2. **Batching**: vLLM automatically batches requests for maximum throughput
3. **Streaming**: Use `stream=true` for better UX in chat applications
4. **Caching**: emptyDir volume caches the model (no re-download)
5. **Multi-GPU**: For 70B+ models, set `tensor_parallel_size=2+`
6. **Fair Benchmarking**: Use the same dtype (float16) for both deployments

## 📝 Extensions

### Use Different Model

Change in `k8s/inference-deployment.yaml`:

```yaml
env:
- name: MODEL_NAME
  value: "meta-llama/Llama-3.2-8B-Instruct"
- name: MAX_MODEL_LEN
  value: "8192"
```

### Prometheus Integration

vLLM exports Prometheus metrics. Add a ServiceMonitor:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-inference
spec:
  selector:
    matchLabels:
      app: llm-inference
  endpoints:
  - port: http
    path: /metrics
```

### Autoscaling (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference-cluster
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔗 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)

## 📄 License

MIT License
