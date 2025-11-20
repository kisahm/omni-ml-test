# LLM Kubernetes Test Application

Eine vollständige LLM-Inference-Anwendung für Kubernetes mit vLLM und GPU-Unterstützung, um Performance-Unterschiede zwischen Cluster- und Edge-Deployments zu messen.

## 🎯 Projektziel

Diese Anwendung ermöglicht es, LLM-Inference auf Kubernetes mit GPUs durchzuführen und die Performance zwischen verschiedenen Deployment-Szenarien zu vergleichen:

- **Cluster Deployment**: Inference-Server im zentralen Kubernetes-Cluster (niedrige Latenz)
- **Edge Deployment**: Inference-Server auf Edge-Nodes in anderen Regionen (hohe Latenz)

## 🏗️ Architektur

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
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Features

- ✅ **vLLM Engine**: Hochoptimierte LLM-Inference mit PagedAttention
- ✅ **OpenAI-kompatible API**: `/v1/chat/completions` und `/v1/completions`
- ✅ **GPU-Beschleunigung**: Echte GPU-Workloads mit CUDA
- ✅ **Streaming Support**: Server-Sent Events (SSE) für Token-Streaming
- ✅ **llmperf-kompatibel**: Benchmark-Metriken (TTFT, Throughput, Latenz)
- ✅ **Performance-Monitoring**: GPU-Auslastung, Token/s, Latenz
- ✅ **Kubernetes-Ready**: Deployments für Cluster und Edge
- ✅ **Vortrainierte Modelle**: Phi-3-mini-4k (3.8B Parameter)

## 🛠️ Technologie-Stack

- **LLM Engine**: vLLM 0.5.4
- **Framework**: PyTorch 2.3.0
- **API**: FastAPI (OpenAI-kompatibel)
- **Container**: Docker (NVIDIA CUDA 12.1)
- **Orchestrierung**: Kubernetes
- **GPU**: NVIDIA GPUs (CUDA)
- **Monitoring**: Prometheus-Client, NVML
- **Modell**: microsoft/Phi-3-mini-4k-instruct (3.8B)

## 📁 Projektstruktur

```
.
├── src/
│   ├── inference.py       # vLLM Inference Service (OpenAI-API)
│   └── metrics.py         # Performance-Metriken
├── k8s/
│   ├── inference-deployment.yaml  # Cluster Inference
│   ├── edge-deployment.yaml       # Edge Inference
│   └── configmap.yaml            # Konfiguration
├── benchmark/
│   └── llm_benchmark.py   # LLM Latenz-Benchmark (llmperf-style)
├── Dockerfile             # vLLM Docker Build
├── requirements.txt       # Python-Abhängigkeiten
├── Makefile              # Build & Deploy Commands
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Kubernetes-Cluster mit GPU-Nodes
- NVIDIA GPU Operator installiert
- Docker Registry Zugriff
- kubectl konfiguriert
- Mindestens 16GB GPU-Memory pro Node

### 2. Image bauen und pushen

```bash
# Image bauen
make build

# Image pushen
make push
```

**Oder manuell:**

```bash
docker build -t kisahm/ml-app:latest .
docker push kisahm/ml-app:latest
```

### 3. GPU Nodes labeln

```bash
# Cluster Node
kubectl label nodes <node-name> \
  accelerator=nvidia-gpu \
  node-location=cluster

# Edge Node (andere Region)
kubectl label nodes <edge-node-name> \
  accelerator=nvidia-gpu \
  node-location=edge
```

### 4. Inference Services deployen

```bash
# Cluster Inference
make deploy-inference

# Edge Inference
make deploy-edge

# Oder beide gleichzeitig
make deploy-all

# Status prüfen
make status
```

Das Deployment:
- Lädt das Phi-3-mini-4k-instruct Modell (ca. 7.5GB)
- Startet vLLM Engine mit GPU-Beschleunigung
- Öffnet OpenAI-kompatible API auf Port 8000
- Start dauert ca. 2-3 Minuten (Model-Download + Initialisierung)

### 5. Services testen

```bash
# Health Check
kubectl get svc llm-inference-cluster-service
CLUSTER_IP=$(kubectl get svc llm-inference-cluster-service -o jsonpath='{.spec.clusterIP}')

curl http://$CLUSTER_IP:8000/health | jq .

# Chat Completion Test
curl -X POST http://$CLUSTER_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .
```

### 6. Benchmark durchführen

```bash
# Automatischer Benchmark
make benchmark

# Oder manuell
python3 benchmark/llm_benchmark.py \
  --cluster-url http://<cluster-service>:8000 \
  --edge-url http://<edge-service>:8000 \
  --requests 50 \
  --max-tokens 256 \
  --mode compare
```

## 📊 API Endpoints

### OpenAI-kompatible API

```bash
# List Models
GET /v1/models

# Chat Completion
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

# Text Completion
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
# Health Check
GET /health

# Metrics
GET /metrics

# Reset Metrics
POST /metrics/reset
```

## 🔍 Benchmark-Metriken (llmperf-style)

### Sequential Benchmark

Testet Anfragen nacheinander:

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 50 \
  --mode sequential
```

**Metriken:**
- Mean/Median/P95/P99 Latenz
- Tokens/Sekunde (Durchsatz)
- GPU-Auslastung

### Concurrent Benchmark

Testet parallele Anfragen:

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 50 \
  --concurrency 5 \
  --mode concurrent
```

### Streaming Benchmark

Testet Time-to-First-Token (TTFT):

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 20 \
  --mode streaming
```

**Streaming-Metriken:**
- **TTFT** (Time to First Token): Zeit bis zum ersten generierten Token
- **Inter-Token Latency**: Durchschnittliche Zeit zwischen Tokens
- **Tokens/Sekunde**: Streaming-Durchsatz

### Cluster vs Edge Comparison

Vergleicht Cluster und Edge Performance:

```bash
python3 benchmark/llm_benchmark.py \
  --cluster-url http://cluster:8000 \
  --edge-url http://edge:8000 \
  --requests 50 \
  --mode compare \
  --output results.json
```

Beispiel-Output:

```
=================================================================
LLM LATENCY COMPARISON: Cluster vs Edge
=================================================================

--- CLUSTER ---
  Success Rate: 100.0%
  Latency:
    Mean:      245.32 ms
    Median:    238.45 ms
    P95:       312.67 ms
    P99:       345.23 ms
  Throughput:
    Mean:      42.5 tokens/sec
    Total tokens: 12,450

--- EDGE ---
  Success Rate: 100.0%
  Latency:
    Mean:      587.91 ms
    Median:    572.12 ms
    P95:       678.34 ms
    P99:       723.45 ms
  Throughput:
    Mean:      38.2 tokens/sec
    Total tokens: 11,234

--- COMPARISON ---
  Cluster Mean Latency: 245.32 ms
  Edge Mean Latency:    587.91 ms
  Difference:           +342.59 ms (+139.6%)

  Cluster Throughput: 42.5 tokens/sec
  Edge Throughput:    38.2 tokens/sec
```

## 🎯 Unterstützte LLM-Modelle

Die Anwendung unterstützt alle vLLM-kompatiblen Modelle. Ändere einfach die `MODEL_NAME` Umgebungsvariable:

### Empfohlene Modelle:

**Kleine Modelle (< 10B, 1x GPU):**
- `microsoft/Phi-3-mini-4k-instruct` (3.8B) - **Default**
- `microsoft/Phi-3-small-8k-instruct` (7B)
- `meta-llama/Llama-3.2-3B-Instruct` (3B)

**Mittlere Modelle (10-20B, 1-2x GPUs):**
- `mistralai/Mistral-7B-Instruct-v0.3` (7B)
- `meta-llama/Llama-3.2-8B-Instruct` (8B)

**Große Modelle (> 20B, 2+ GPUs):**
- `meta-llama/Meta-Llama-3-70B-Instruct` (70B)

Modell ändern:

```yaml
# In k8s/inference-deployment.yaml oder k8s/edge-deployment.yaml
env:
- name: MODEL_NAME
  value: "mistralai/Mistral-7B-Instruct-v0.3"
- name: MAX_MODEL_LEN
  value: "8192"
```

## 🔧 Konfiguration

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
```

### Resource Requests

Standardmäßig:

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

Passe diese nach Bedarf für größere Modelle an (z.B. 2-4 GPUs für 70B Modelle).

## 📈 Monitoring

### GPU-Metriken abrufen

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

### Logs anschauen

```bash
# Cluster Logs
make logs-inference

# Edge Logs
make logs-edge

# Oder direkt mit kubectl
kubectl logs -l app=llm-inference,location=cluster -f
```

## 🧪 Lokales Testen (ohne Kubernetes)

```bash
# Dependencies installieren
pip install -r requirements.txt

# Inference Server starten (benötigt GPU)
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct python src/inference.py

# In einem anderen Terminal testen
curl http://localhost:8000/health

# Chat Completion testen
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi!"}],"max_tokens":50}' \
  | jq .
```

## 🐛 Troubleshooting

### Pods starten nicht

```bash
# Pod Status prüfen
kubectl get pods -l app=llm-inference

# Events prüfen
kubectl describe pod <pod-name>

# Logs prüfen
kubectl logs <pod-name>
```

### GPU nicht gefunden

```bash
# GPU Nodes prüfen
kubectl get nodes -o json | jq '.items[].status.allocatable | select(."nvidia.com/gpu" != null)'

# NVIDIA Device Plugin
kubectl get ds -n kube-system nvidia-device-plugin-daemonset

# GPU Operator
kubectl get pods -n gpu-operator-resources
```

### Model Download fehlschlägt

Das Modell wird beim Start automatisch von HuggingFace heruntergeladen. Bei Problemen:

```bash
# HuggingFace Token setzen (für gated models)
kubectl create secret generic hf-secret \
  --from-literal=HF_TOKEN=your_token

# Und in deployment.yaml hinzufügen:
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-secret
      key: HF_TOKEN
```

### Out of Memory (OOM)

Für größere Modelle:
- Erhöhe `memory` Limits in Deployments
- Reduziere `MAX_MODEL_LEN`
- Verwende kleineres Modell
- Nutze Tensor Parallelism (`tensor_parallel_size`)

## 💡 Tipps

1. **Model wählen**: Phi-3-mini ist perfekt für Tests (schnell, klein, gut)
2. **Batching**: vLLM batcht automatisch Anfragen für maximalen Durchsatz
3. **Streaming**: Nutze `stream=true` für bessere UX bei Chat-Anwendungen
4. **Caching**: emptyDir Volume cacht das Modell (kein erneuter Download)
5. **Multi-GPU**: Für 70B+ Modelle setze `tensor_parallel_size=2+`

## 📝 Erweitungen

### Anderes Modell verwenden

Ändere in `k8s/inference-deployment.yaml`:

```yaml
env:
- name: MODEL_NAME
  value: "meta-llama/Llama-3.2-8B-Instruct"
- name: MAX_MODEL_LEN
  value: "8192"
```

### Prometheus-Integration

vLLM exportiert Prometheus-Metriken. Füge einen ServiceMonitor hinzu:

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

## 🔗 Ressourcen

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)

## 📄 Lizenz

MIT License
