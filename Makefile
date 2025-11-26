.PHONY: help build push deploy-inference deploy-edge deploy-all rollout clean benchmark benchmark-deps status buildx-setup

# Configuration
IMAGE_NAME ?= ml-app
IMAGE_TAG ?= latest
REGISTRY ?= kisahm
FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
PLATFORM ?= linux/amd64

help:
	@echo "LLM Kubernetes Test Application - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  buildx-setup       - Setup Docker Buildx for cross-platform builds"
	@echo "  build              - Build Docker image for AMD64 (x86_64)"
	@echo "  push               - Build and push Docker image to registry"
	@echo "  deploy-inference   - Deploy LLM inference service (cluster)"
	@echo "  deploy-edge        - Deploy LLM inference service (edge)"
	@echo "  deploy-all         - Deploy both cluster and edge inference"
	@echo "  rollout            - Restart deployments to pull new image"
	@echo "  clean              - Clean up Kubernetes resources"
	@echo "  benchmark-deps     - Install benchmark dependencies"
	@echo "  benchmark          - Run LLM latency benchmark"
	@echo "  status             - Show deployment status"
	@echo "  logs-inference     - Show inference logs (cluster)"
	@echo "  logs-edge          - Show inference logs (edge)"
	@echo ""
	@echo "Building on Apple Silicon (M1/M2/M3):"
	@echo "  The images are built for linux/amd64 (x86_64) by default"
	@echo "  Run 'make buildx-setup' once to configure buildx"
	@echo ""

buildx-setup:
	@echo "Setting up Docker Buildx for cross-platform builds..."
	@docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch
	@docker buildx inspect --bootstrap
	@echo "✓ Buildx setup complete!"

build: buildx-setup
	@echo "Building Docker image for $(PLATFORM): $(FULL_IMAGE)"
	docker buildx build \
		--platform $(PLATFORM) \
		-t $(FULL_IMAGE) \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		--load \
		.
	@echo "✓ Image built successfully for $(PLATFORM)"

push: buildx-setup
	@echo "Building and pushing Docker image for $(PLATFORM): $(FULL_IMAGE)"
	docker buildx build \
		--platform $(PLATFORM) \
		-t $(FULL_IMAGE) \
		-t $(REGISTRY)/$(IMAGE_NAME):latest \
		--push \
		.
	@echo "✓ Image pushed successfully to $(REGISTRY)/$(IMAGE_NAME):latest"

deploy-inference:
	@echo "Deploying LLM inference service (cluster)..."
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/inference-deployment.yaml

deploy-edge:
	@echo "Deploying LLM inference service (edge)..."
	kubectl apply -f k8s/edge-deployment.yaml

deploy-all: deploy-inference deploy-edge
	@echo "All LLM inference services deployed!"

rollout:
	@echo "Restarting deployments to pull new image..."
	kubectl rollout restart deployment llm-inference-cluster
	kubectl rollout restart deployment llm-inference-edge -n edge
	@echo "Waiting for rollout to complete..."
	kubectl rollout status deployment llm-inference-cluster
	kubectl rollout status deployment llm-inference-edge -n edge
	@echo "✓ Rollout complete!"

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
	kubectl logs -n edge -l app=llm-inference,location=edge -f

status:
	@echo "=== LLM Inference Deployments (Cluster) ==="
	kubectl get deployments -l app=llm-inference
	@echo ""
	@echo "=== LLM Inference Deployments (Edge) ==="
	kubectl get deployments -n edge -l app=llm-inference
	@echo ""
	@echo "=== Services (Cluster) ==="
	kubectl get services -l app=llm-inference
	@echo ""
	@echo "=== Services (Edge) ==="
	kubectl get services -n edge -l app=llm-inference
	@echo ""
	@echo "=== Pods (Cluster) ==="
	kubectl get pods -l app=llm-inference
	@echo ""
	@echo "=== Pods (Edge) ==="
	kubectl get pods -n edge -l app=llm-inference

benchmark-deps:
	@echo "Checking benchmark dependencies..."
	@pip3 install -q --break-system-packages -r benchmark/requirements.txt
	@echo "✓ Benchmark dependencies installed"

benchmark: benchmark-deps
	@echo "Running LLM benchmark..."
	@echo "Getting service URLs..."
	$(eval CLUSTER_URL := $(shell kubectl get svc llm-inference-cluster-service -o jsonpath='{.spec.clusterIP}'))
	$(eval EDGE_URL := $(shell kubectl get svc llm-inference-edge-service -n edge -o jsonpath='{.spec.clusterIP}'))
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
