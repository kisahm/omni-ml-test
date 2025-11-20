#!/bin/bash
# Deployment Script for ML Kubernetes Application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${IMAGE_NAME:-ml-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-your-registry}"
NAMESPACE="${NAMESPACE:-default}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}ML Kubernetes Deployment Script${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Configuration:"
echo "  Image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Namespace: ${NAMESPACE}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install kubectl."
    exit 1
fi
print_status "kubectl found"

if ! command -v docker &> /dev/null; then
    print_warning "docker not found. Skipping build step."
    SKIP_BUILD=true
else
    print_status "docker found"
fi

# Build and push image
if [ "$SKIP_BUILD" != "true" ]; then
    echo ""
    echo "Building Docker image..."
    docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
    print_status "Image built"

    echo ""
    echo "Pushing Docker image..."
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    print_status "Image pushed"
fi

# Update image in manifests
echo ""
echo "Updating image in Kubernetes manifests..."
for file in k8s/*.yaml; do
    if grep -q "image: your-registry" "$file"; then
        sed -i.bak "s|image: your-registry/ml-app:latest|image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" "$file"
        rm "${file}.bak"
        print_status "Updated $file"
    fi
done

# Deploy to Kubernetes
echo ""
echo "Deploying to Kubernetes..."

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
print_status "Namespace ready"

# Apply ConfigMap
echo ""
echo "Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml -n ${NAMESPACE}
print_status "ConfigMap applied"

# Apply PVCs
echo ""
echo "Creating PersistentVolumeClaims..."
kubectl apply -f k8s/training-job.yaml -n ${NAMESPACE} | grep -i persistentvolumeclaim || true
print_status "PVCs created"

# Deploy based on user choice
echo ""
echo "What would you like to deploy?"
echo "  1) Training Job only"
echo "  2) Inference (Cluster) only"
echo "  3) Inference (Edge) only"
echo "  4) All (Training + Inference Cluster + Edge)"
echo "  5) Skip deployment"
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        kubectl apply -f k8s/training-job.yaml -n ${NAMESPACE}
        print_status "Training job deployed"
        ;;
    2)
        kubectl apply -f k8s/inference-deployment.yaml -n ${NAMESPACE}
        print_status "Cluster inference deployed"
        ;;
    3)
        kubectl apply -f k8s/edge-deployment.yaml -n ${NAMESPACE}
        print_status "Edge inference deployed"
        ;;
    4)
        kubectl apply -f k8s/training-job.yaml -n ${NAMESPACE}
        kubectl apply -f k8s/inference-deployment.yaml -n ${NAMESPACE}
        kubectl apply -f k8s/edge-deployment.yaml -n ${NAMESPACE}
        print_status "All components deployed"
        ;;
    5)
        print_warning "Skipping deployment"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Show status
echo ""
echo "Deployment Status:"
echo ""
echo "Jobs:"
kubectl get jobs -n ${NAMESPACE} -l app=ml-training 2>/dev/null || echo "  No training jobs found"
echo ""
echo "Deployments:"
kubectl get deployments -n ${NAMESPACE} -l app=ml-inference 2>/dev/null || echo "  No deployments found"
echo ""
echo "Services:"
kubectl get services -n ${NAMESPACE} -l app=ml-inference 2>/dev/null || echo "  No services found"
echo ""
echo "Pods:"
kubectl get pods -n ${NAMESPACE} -l app=ml-inference 2>/dev/null || kubectl get pods -n ${NAMESPACE} -l app=ml-training 2>/dev/null || echo "  No pods found"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Next steps:"
echo "  - Check logs: kubectl logs -n ${NAMESPACE} -l app=ml-training -f"
echo "  - Run benchmark: make benchmark"
echo "  - Test inference: curl http://<service-ip>:8000/health"
