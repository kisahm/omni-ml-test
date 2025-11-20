# Quick Start Guide

## 1. Setup (5 Minuten)

### Node Labels setzen

```bash
# GPU Nodes markieren
kubectl label nodes <gpu-node-1> accelerator=nvidia-gpu node-location=cluster
kubectl label nodes <gpu-node-2> accelerator=nvidia-gpu node-location=edge
```

### Image Registry konfigurieren

In allen `k8s/*.yaml` Dateien:
- Ersetze `your-registry` mit deiner Registry (z.B. `docker.io/username`)

Oder nutze das Deploy-Script:

```bash
export REGISTRY=docker.io/username
export IMAGE_NAME=ml-app
./deploy.sh
```

## 2. Training (10 Minuten)

```bash
# Manuell
kubectl apply -f k8s/training-job.yaml

# Oder mit Make
make deploy-train

# Logs verfolgen
kubectl logs -l app=ml-training -f
```

Wartet bis das Training fertig ist (Status: Completed).

## 3. Inference deployen (2 Minuten)

```bash
# Cluster Inference
kubectl apply -f k8s/inference-deployment.yaml

# Edge Inference
kubectl apply -f k8s/edge-deployment.yaml

# Status prüfen
kubectl get pods -l app=ml-inference
```

## 4. Services testen (1 Minute)

```bash
# Service IP holen
CLUSTER_IP=$(kubectl get svc ml-inference-cluster-service -o jsonpath='{.spec.clusterIP}')
EDGE_IP=$(kubectl get svc ml-inference-edge-service -o jsonpath='{.spec.clusterIP}')

# Health Check
curl http://$CLUSTER_IP:8000/health | jq .

# Test-Bild erstellen und hochladen
python3 -c "from PIL import Image; import numpy as np; Image.fromarray(np.random.randint(0,255,(32,32,3), dtype=np.uint8)).save('test.png')"

curl -X POST -F "file=@test.png" http://$CLUSTER_IP:8000/predict | jq .
```

## 5. Benchmark durchführen (5 Minuten)

```bash
python3 benchmark/benchmark.py \
  --cluster-url http://$CLUSTER_IP:8000 \
  --edge-url http://$EDGE_IP:8000 \
  --requests 100 \
  --mode compare
```

## Lokales Testen (ohne Kubernetes)

```bash
# Training
python src/train.py --epochs 2 --batch-size 64

# Inference
python src/inference.py &

# Test
curl http://localhost:8000/health
```

## Mit Docker Compose

```bash
# Training
docker-compose --profile training up training

# Inference
docker-compose up inference
```

## Troubleshooting

### GPU nicht gefunden?

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable | select(."nvidia.com/gpu" != null)'
```

### Pod startet nicht?

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Training schlägt fehl?

Prüfe GPU-Verfügbarkeit und Resources in `k8s/training-job.yaml`.

## Wichtige Befehle

```bash
# Status
kubectl get all -l app=ml-inference

# Logs
kubectl logs -l app=ml-training -f
kubectl logs -l app=ml-inference,location=cluster -f

# Aufräumen
kubectl delete -f k8s/
```

## Nächste Schritte

1. Anpassen der Node-Selectors in `k8s/` Dateien
2. Resource-Limits anpassen
3. Prometheus-Integration für Monitoring
4. Automatisiertes CI/CD Setup
5. Multi-Region Deployment

Siehe [README.md](README.md) für vollständige Dokumentation.
