.PHONY: help build push deploy-train deploy-inference deploy-edge clean benchmark

# Configuration
IMAGE_NAME ?= ml-app
IMAGE_TAG ?= latest
REGISTRY ?= kisahm
FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

help:
	@echo "ML Kubernetes Test Application - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build              - Build Docker image"
	@echo "  push               - Push Docker image to registry"
	@echo "  deploy-train       - Deploy training job"
	@echo "  deploy-inference   - Deploy inference service (cluster)"
	@echo "  deploy-edge        - Deploy inference service (edge)"
	@echo "  deploy-all         - Deploy everything"
	@echo "  clean              - Clean up Kubernetes resources"
	@echo "  benchmark          - Run latency benchmark"
	@echo "  logs-train         - Show training logs"
	@echo "  logs-inference     - Show inference logs"
	@echo ""

build:
	@echo "Building Docker image: $(FULL_IMAGE)"
	docker build -t $(FULL_IMAGE) .
	docker tag $(FULL_IMAGE) $(REGISTRY)/$(IMAGE_NAME):latest

push: build
	@echo "Pushing Docker image: $(FULL_IMAGE)"
	docker push $(FULL_IMAGE)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

deploy-pvcs:
	@echo "Creating PersistentVolumeClaims..."
	kubectl apply -f k8s/training-job.yaml | grep -i persistentvolumeclaim || true

deploy-train: deploy-pvcs
	@echo "Deploying training job..."
	kubectl apply -f k8s/training-job.yaml

deploy-inference:
	@echo "Deploying inference service (cluster)..."
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/inference-deployment.yaml

deploy-edge:
	@echo "Deploying inference service (edge)..."
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/edge-deployment.yaml

deploy-all: deploy-train deploy-inference deploy-edge
	@echo "All components deployed!"

clean:
	@echo "Cleaning up Kubernetes resources..."
	kubectl delete -f k8s/training-job.yaml --ignore-not-found=true
	kubectl delete -f k8s/inference-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/edge-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/configmap.yaml --ignore-not-found=true

logs-train:
	@echo "Fetching training logs..."
	kubectl logs -l app=ml-training -f

logs-inference:
	@echo "Fetching inference logs (cluster)..."
	kubectl logs -l app=ml-inference,location=cluster -f

logs-edge:
	@echo "Fetching inference logs (edge)..."
	kubectl logs -l app=ml-inference,location=edge -f

status:
	@echo "=== Training Jobs ==="
	kubectl get jobs -l app=ml-training
	@echo ""
	@echo "=== Inference Deployments ==="
	kubectl get deployments -l app=ml-inference
	@echo ""
	@echo "=== Services ==="
	kubectl get services -l app=ml-inference
	@echo ""
	@echo "=== Pods ==="
	kubectl get pods -l app=ml-inference

benchmark:
	@echo "Running benchmark..."
	@echo "Getting service URLs..."
	$(eval CLUSTER_URL := $(shell kubectl get svc ml-inference-cluster-service -o jsonpath='{.spec.clusterIP}'))
	$(eval EDGE_URL := $(shell kubectl get svc ml-inference-edge-service -o jsonpath='{.spec.clusterIP}'))
	python3 benchmark/benchmark.py \
		--cluster-url http://$(CLUSTER_URL):8000 \
		--edge-url http://$(EDGE_URL):8000 \
		--requests 100 \
		--mode compare

# Local development
run-train-local:
	python3 src/train.py --epochs 2 --batch-size 64

run-inference-local:
	python3 src/inference.py

test-inference-local:
	@echo "Testing local inference..."
	curl -X POST http://localhost:8000/health | jq .
