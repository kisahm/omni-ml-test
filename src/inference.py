"""
ML Inference Service with FastAPI
Provides REST API for model inference with performance tracking
"""
import os
import io
import time
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Optional

from model import create_model, get_model_info
from metrics import MetricsCollector, LatencyTracker

# CIFAR-10 classes
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Global variables
app = FastAPI(title="ML Inference Service", version="1.0.0")
model = None
device = None
metrics_collector = None
latency_tracker = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[Dict[str, float]]
    top_class: str
    confidence: float
    latency_ms: float
    device: str
    gpu_metrics: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    device: str
    gpu_available: bool
    model_loaded: bool
    metrics: Dict


class MetricsResponse(BaseModel):
    """Response model for metrics"""
    system_metrics: Dict
    latency_stats: Dict


def load_model(model_path: str = 'model.pth'):
    """Load the trained model"""
    global model, device, metrics_collector, latency_tracker

    print("Initializing inference service...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize metrics
    metrics_collector = MetricsCollector()
    latency_tracker = LatencyTracker()

    # Create model
    model = create_model(device)

    # Load weights if available
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded! Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")

    model.eval()
    print("Inference service ready!")


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = os.environ.get('MODEL_PATH', 'model.pth')
    load_model(model_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_metrics = metrics_collector.get_gpu_metrics() if metrics_collector else {}

    return HealthResponse(
        status="healthy" if model is not None else "initializing",
        device=str(device),
        gpu_available=torch.cuda.is_available(),
        model_loaded=model is not None,
        metrics=gpu_metrics
    )


@app.get("/info")
async def model_info():
    """Get model information"""
    info = get_model_info()
    info['device'] = str(device)
    info['gpu_available'] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image

    Performs inference on CIFAR-10 model with GPU acceleration
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and preprocess image
        contents = await file.read()
        image_tensor = preprocess_image(contents)
        image_tensor = image_tensor.to(device)

        # Inference with timing
        start_time = time.time()

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Synchronize GPU for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()

        latency_ms = (time.time() - start_time) * 1000

        # Track latency
        latency_tracker.record(latency_ms)

        # Get predictions
        probs = probabilities[0].cpu().numpy()
        predictions = [
            {"class": CLASSES[i], "probability": float(probs[i])}
            for i in range(len(CLASSES))
        ]
        predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)

        top_class = predictions[0]['class']
        confidence = predictions[0]['probability']

        # Collect metrics
        gpu_metrics = metrics_collector.get_gpu_metrics()

        return PredictionResponse(
            predictions=predictions,
            top_class=top_class,
            confidence=confidence,
            latency_ms=latency_ms,
            device=str(device),
            gpu_metrics=gpu_metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple images
    Processes images in a single batch for better GPU utilization
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess all images
        image_tensors = []
        for file in files:
            contents = await file.read()
            tensor = preprocess_image(contents)
            image_tensors.append(tensor)

        # Stack into batch
        batch_tensor = torch.cat(image_tensors, dim=0).to(device)

        # Batch inference with timing
        start_time = time.time()

        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        latency_ms = (time.time() - start_time) * 1000
        latency_per_image = latency_ms / len(files)

        # Process results
        results = []
        for i in range(len(files)):
            probs = probabilities[i].cpu().numpy()
            predictions = [
                {"class": CLASSES[j], "probability": float(probs[j])}
                for j in range(len(CLASSES))
            ]
            predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)

            results.append({
                "filename": files[i].filename,
                "top_class": predictions[0]['class'],
                "confidence": predictions[0]['probability'],
                "predictions": predictions[:3]  # Top 3
            })

        return {
            "batch_size": len(files),
            "total_latency_ms": latency_ms,
            "latency_per_image_ms": latency_per_image,
            "throughput_images_per_sec": 1000 / latency_per_image,
            "results": results,
            "device": str(device)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system and performance metrics"""
    system_metrics = metrics_collector.get_all_metrics()
    latency_stats = latency_tracker.get_stats()

    return MetricsResponse(
        system_metrics=system_metrics,
        latency_stats=latency_stats
    )


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset latency statistics"""
    latency_tracker.reset()
    return {"status": "metrics reset"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ML Inference API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8000))

    print(f"Starting inference service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
