.PHONY: help build push deploy-inference deploy-edge deploy-all clean benchmark status

# Configuration
IMAGE_NAME ?= ml-app
IMAGE_TAG ?= latest
REGISTRY ?= kisahm
FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

help:
	@echo "LLM Kubernetes Test Application - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build              - Build Docker image"
	@echo "  push               - Push Docker image to registry"
	@echo "  deploy-inference   - Deploy LLM inference service (cluster)"
	@echo "  deploy-edge        - Deploy LLM inference service (edge)"
	@echo "  deploy-all         - Deploy both cluster and edge inference"
	@echo "  clean              - Clean up Kubernetes resources"
	@echo "  benchmark          - Run LLM latency benchmark"
	@echo "  status             - Show deployment status"
	@echo "  logs-inference     - Show inference logs (cluster)"
	@echo "  logs-edge          - Show inference logs (edge)"
	@echo ""

build:
	@echo "Building Docker image: $(FULL_IMAGE)"
	docker build -t $(FULL_IMAGE) .
	docker tag $(FULL_IMAGE) $(REGISTRY)/$(IMAGE_NAME):latest

push: build
	@echo "Pushing Docker image: $(FULL_IMAGE)"
	docker push $(FULL_IMAGE)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

deploy-inference:
	@echo "Deploying LLM inference service (cluster)..."
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/inference-deployment.yaml

deploy-edge:
	@echo "Deploying LLM inference service (edge)..."
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/edge-deployment.yaml

deploy-all: deploy-inference deploy-edge
	@echo "All LLM inference services deployed!"

clean:
	@echo "Cleaning up Kubernetes resources..."
	kubectl delete -f k8s/inference-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/edge-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/configmap.yaml --ignore-not-found=true

logs-inference:
	@echo "Fetching LLM inference logs (cluster)..."
	kubectl logs -l app=llm-inference,location=cluster -f

logs-edge:
	@echo "Fetching LLM inference logs (edge)..."
	kubectl logs -l app=llm-inference,location=edge -f

status:
	@echo "=== LLM Inference Deployments ==="
	kubectl get deployments -l app=llm-inference
	@echo ""
	@echo "=== Services ==="
	kubectl get services -l app=llm-inference
	@echo ""
	@echo "=== Pods ==="
	kubectl get pods -l app=llm-inference

benchmark:
	@echo "Running LLM benchmark..."
	@echo "Getting service URLs..."
	$(eval CLUSTER_URL := $(shell kubectl get svc llm-inference-cluster-service -o jsonpath='{.spec.clusterIP}'))
	$(eval EDGE_URL := $(shell kubectl get svc llm-inference-edge-service -o jsonpath='{.spec.clusterIP}'))
	python3 benchmark/llm_benchmark.py \
		--cluster-url http://$(CLUSTER_URL):8000 \
		--edge-url http://$(EDGE_URL):8000 \
		--requests 50 \
		--max-tokens 256 \
		--mode compare

# Local development
run-inference-local:
	MODEL_NAME=microsoft/Phi-3-mini-4k-instruct python3 src/inference.py

test-inference-local:
	@echo "Testing local LLM inference..."
	curl -X POST http://localhost:8000/health | jq .
	@echo ""
	@echo "Testing chat completion..."
	curl -X POST http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}' \
		| jq .
