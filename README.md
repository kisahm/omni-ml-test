# ML Kubernetes Test Application

Eine vollständige ML-Test-Anwendung für Kubernetes mit GPU-Unterstützung, um Performance-Unterschiede zwischen Cluster- und Edge-Deployments zu messen.

## 🎯 Projektziel

Diese Anwendung ermöglicht es, ML Training und Inference auf Kubernetes mit GPUs durchzuführen und die Performance zwischen verschiedenen Deployment-Szenarien zu vergleichen:

- **Cluster Deployment**: Inference-Server im zentralen Kubernetes-Cluster
- **Edge Deployment**: Inference-Server auf Edge-Nodes in anderen Regionen mit hoher Latenz

## 🏗️ Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                               │
│  ┌──────────────┐      ┌────────────────────────────┐       │
│  │  Training    │      │   Inference (Cluster)      │       │
│  │  Job         │──────▶   - 2 Replicas             │       │
│  │  (GPU)       │      │   - GPU Support            │       │
│  └──────────────┘      │   - Low Latency            │       │
│                        └────────────────────────────┘       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ High Latency Network
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Edge Location                             │
│                                                               │
│  ┌────────────────────────────┐                             │
│  │   Inference (Edge)         │                             │
│  │   - 2 Replicas             │                             │
│  │   - GPU Support            │                             │
│  │   - High Latency           │                             │
│  └────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Features

- ✅ **Echte GPU-Workloads**: ResNet18 Training auf CIFAR-10
- ✅ **REST API**: FastAPI-basierter Inference-Service
- ✅ **Performance-Monitoring**: GPU-Auslastung, Latenz, Durchsatz
- ✅ **Batch-Inference**: Optimierte Batch-Verarbeitung
- ✅ **Kubernetes-Ready**: Deployments für Cluster und Edge
- ✅ **Benchmark-Tools**: Automatisierte Latenz-Vergleiche
- ✅ **Metriken**: Detaillierte Performance-Statistiken

## 🛠️ Technologie-Stack

- **ML Framework**: PyTorch 2.1.0
- **API Framework**: FastAPI
- **Container**: Docker (NVIDIA CUDA 12.1)
- **Orchestrierung**: Kubernetes
- **GPU**: NVIDIA GPUs (CUDA)
- **Monitoring**: Prometheus-Client, NVML

## 📁 Projektstruktur

```
.
├── src/
│   ├── model.py           # ResNet18 Modell-Definition
│   ├── train.py           # Training-Script mit GPU
│   ├── inference.py       # Inference-API (FastAPI)
│   └── metrics.py         # Performance-Metriken
├── k8s/
│   ├── training-job.yaml  # Training Job Manifest
│   ├── inference-deployment.yaml  # Cluster Inference
│   ├── edge-deployment.yaml       # Edge Inference
│   └── configmap.yaml     # Konfiguration
├── benchmark/
│   ├── benchmark.py       # Latenz-Benchmark-Tool
│   └── create_test_images.py  # Test-Bilder erstellen
├── Dockerfile             # Multi-Stage Docker Build
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

### 2. Image bauen und pushen

```bash
# Image bauen
make build IMAGE_NAME=ml-app REGISTRY=your-registry

# Image pushen
make push IMAGE_NAME=ml-app REGISTRY=your-registry
```

**Oder manuell:**

```bash
docker build -t your-registry/ml-app:latest .
docker push your-registry/ml-app:latest
```

### 3. Kubernetes Manifeste anpassen

Bearbeite die YAML-Dateien in `k8s/` und ersetze:
- `your-registry/ml-app:latest` mit deinem Image
- Storage-Klassen nach Bedarf
- Node-Labels/Selectors für dein Cluster

### 4. Training starten

```bash
# PVCs erstellen
kubectl apply -f k8s/training-job.yaml

# Training Job starten
kubectl apply -f k8s/training-job.yaml

# Logs verfolgen
kubectl logs -l app=ml-training -f
```

Das Training:
- Lädt CIFAR-10 Dataset herunter (ca. 170 MB)
- Trainiert ResNet18 für 10 Epochen (ca. 5-10 Minuten mit GPU)
- Speichert das Modell in `/models/model.pth`

### 5. Inference Services deployen

```bash
# Cluster Inference
make deploy-inference

# Edge Inference
make deploy-edge

# Oder beide gleichzeitig
make deploy-all
```

### 6. Services testen

```bash
# Health Check
kubectl get svc ml-inference-cluster-service
CLUSTER_IP=$(kubectl get svc ml-inference-cluster-service -o jsonpath='{.spec.clusterIP}')

curl http://$CLUSTER_IP:8000/health | jq .

# Inference Test (mit Test-Bild)
curl -X POST \
  -F "file=@test_image.png" \
  http://$CLUSTER_IP:8000/predict | jq .
```

### 7. Benchmark durchführen

```bash
# Automatischer Benchmark
make benchmark

# Oder manuell
python3 benchmark/benchmark.py \
  --cluster-url http://<cluster-service>:8000 \
  --edge-url http://<edge-service>:8000 \
  --requests 100 \
  --mode compare
```

## 📊 API Endpoints

### Inference API

```bash
# Health Check
GET /health

# Model Info
GET /info

# Single Prediction
POST /predict
Content-Type: multipart/form-data
Body: file=<image>

# Batch Prediction
POST /predict/batch
Content-Type: multipart/form-data
Body: files=<image1>, files=<image2>, ...

# Metrics
GET /metrics

# Reset Metrics
POST /metrics/reset
```

### Beispiel: Single Prediction

```bash
curl -X POST \
  -F "file=@image.png" \
  http://localhost:8000/predict

Response:
{
  "predictions": [
    {"class": "cat", "probability": 0.85},
    {"class": "dog", "probability": 0.10},
    ...
  ],
  "top_class": "cat",
  "confidence": 0.85,
  "latency_ms": 12.5,
  "device": "cuda:0",
  "gpu_metrics": {
    "gpu_memory_allocated_mb": 450.2,
    "gpu_utilization_percent": 78
  }
}
```

### Beispiel: Batch Prediction

```bash
curl -X POST \
  -F "files=@img1.png" \
  -F "files=@img2.png" \
  -F "files=@img3.png" \
  http://localhost:8000/predict/batch

Response:
{
  "batch_size": 3,
  "total_latency_ms": 25.3,
  "latency_per_image_ms": 8.4,
  "throughput_images_per_sec": 119.0,
  "results": [...]
}
```

## 🔍 Benchmark-Szenarien

### 1. Sequential Benchmark

Testet Einzelanfragen nacheinander:

```bash
python3 benchmark/benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 100 \
  --mode sequential
```

### 2. Concurrent Benchmark

Testet parallele Anfragen:

```bash
python3 benchmark/benchmark.py \
  --cluster-url http://cluster:8000 \
  --requests 100 \
  --concurrency 10 \
  --mode concurrent
```

### 3. Batch Benchmark

Testet verschiedene Batch-Größen:

```bash
python3 benchmark/benchmark.py \
  --cluster-url http://cluster:8000 \
  --mode batch
```

### 4. Cluster vs Edge Comparison

Vergleicht Cluster und Edge Performance:

```bash
python3 benchmark/benchmark.py \
  --cluster-url http://cluster:8000 \
  --edge-url http://edge:8000 \
  --requests 100 \
  --mode compare \
  --output results.json
```

Beispiel-Output:

```
=================================================================
LATENCY COMPARISON: Cluster vs Edge
=================================================================

--- CLUSTER ---
  Success Rate: 100.0%
  Total Latency:
    Mean:      15.32 ms
    Median:    14.89 ms
    P95:       22.45 ms
    P99:       28.12 ms
  GPU Inference: 8.50 ms
  Network: 6.82 ms

--- EDGE ---
  Success Rate: 100.0%
  Total Latency:
    Mean:      145.67 ms
    Median:    142.33 ms
    P95:       178.90 ms
    P99:       195.23 ms
  GPU Inference: 8.45 ms
  Network: 137.22 ms

--- COMPARISON ---
  Cluster Mean Latency: 15.32 ms
  Edge Mean Latency:    145.67 ms
  Difference:           +130.35 ms (+850.7%)
  → Edge is SLOWER by 130.35 ms
```

## 🎯 Node Labels für Deployment

Für korrektes Deployment benötigst du diese Node-Labels:

```bash
# GPU Nodes markieren
kubectl label nodes <node-name> accelerator=nvidia-gpu

# Cluster Nodes (zentral)
kubectl label nodes <node-name> node-location=cluster

# Edge Nodes (remote)
kubectl label nodes <node-name> node-location=edge
```

Optional: Taints für Edge Nodes

```bash
kubectl taint nodes <edge-node> node-role.kubernetes.io/edge=true:NoSchedule
```

## 🔧 Konfiguration

### Environment Variables

```bash
# Training
EPOCHS=10
BATCH_SIZE=128
LEARNING_RATE=0.1

# Inference
MODEL_PATH=/models/model.pth
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
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
```

Passe diese nach Bedarf in den YAML-Dateien an.

## 📈 Monitoring

### GPU-Metriken abrufen

```bash
# Via API
curl http://<service>:8000/metrics | jq .

# Output:
{
  "system_metrics": {
    "gpu_available": true,
    "gpu_memory_allocated_mb": 512.5,
    "gpu_utilization_percent": 75,
    "gpu_temperature_c": 68
  },
  "latency_stats": {
    "count": 1000,
    "mean_ms": 12.5,
    "p95_ms": 18.2,
    "p99_ms": 22.1
  }
}
```

### Logs anschauen

```bash
# Training Logs
kubectl logs -l app=ml-training -f

# Inference Logs (Cluster)
kubectl logs -l app=ml-inference,location=cluster -f

# Inference Logs (Edge)
kubectl logs -l app=ml-inference,location=edge -f
```

## 🧪 Lokales Testen (ohne Kubernetes)

### Training lokal ausführen

```bash
# Dependencies installieren
pip install -r requirements.txt

# Training starten
python src/train.py --epochs 2 --batch-size 64
```

### Inference lokal starten

```bash
# Inference Server starten
python src/inference.py

# In einem anderen Terminal testen
curl http://localhost:8000/health
```

## 🐛 Troubleshooting

### Training Job hängt

```bash
# Status prüfen
kubectl describe job ml-training-job

# Events prüfen
kubectl get events --sort-by='.lastTimestamp'

# Logs prüfen
kubectl logs -l app=ml-training
```

### GPU nicht gefunden

```bash
# GPU Operator Status
kubectl get pods -n gpu-operator-resources

# Node GPU Status
kubectl describe node <node-name> | grep -i gpu

# NVIDIA Device Plugin
kubectl get ds -n kube-system nvidia-device-plugin-daemonset
```

### Inference Service nicht erreichbar

```bash
# Pod Status
kubectl get pods -l app=ml-inference

# Service Status
kubectl get svc -l app=ml-inference

# Logs prüfen
kubectl logs -l app=ml-inference --tail=50
```

### Hohe Latenz

Mögliche Ursachen:
- Network-Latenz zwischen Nodes
- GPU nicht verfügbar (CPU Fallback)
- Unzureichende Resources
- Throttling

```bash
# GPU Usage in Pod prüfen
kubectl exec -it <pod-name> -- nvidia-smi

# Resource Usage prüfen
kubectl top pods -l app=ml-inference
```

## 📝 Erweitungen & Anpassungen

### Anderes Modell verwenden

Bearbeite `src/model.py`:

```python
def create_model(device='cuda'):
    model = YourCustomModel()
    model = model.to(device)
    return model
```

### Anderes Dataset

Bearbeite `src/train.py` und ändere die Data Loaders.

### Andere Metriken

Erweitere `src/metrics.py` um zusätzliche Metriken.

## 🤝 Contributing

Contributions willkommen! Bitte:
1. Fork das Repository
2. Feature Branch erstellen
3. Changes committen
4. Pull Request erstellen

## 📄 Lizenz

MIT License - siehe LICENSE Datei

## 🔗 Ressourcen

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)

## 💡 Tipps

1. **Training beschleunigen**: Erhöhe `batch_size` und nutze mehrere GPUs
2. **Inference optimieren**: Nutze Batch-Endpoints für höheren Durchsatz
3. **Latenz reduzieren**: Deploy Inference nah am Client (Edge)
4. **Kosten sparen**: Nutze Spot/Preemptible Instances für Training
5. **Monitoring**: Integriere Prometheus für langfristiges Monitoring

## 📧 Support

Bei Fragen oder Problemen:
- Issue erstellen im Repository
- Logs und Konfiguration bereitstellen
- Kubernetes & GPU Versionen angeben
