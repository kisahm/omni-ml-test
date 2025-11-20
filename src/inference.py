"""
LLM Inference Service with vLLM and FastAPI
Provides OpenAI-compatible API for LLM inference with performance tracking
"""
import os
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, AsyncIterator
import asyncio
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.utils import random_uuid

from metrics import MetricsCollector, LatencyTracker

# Global variables
app = FastAPI(title="LLM Inference Service (vLLM)", version="1.0.0")
engine: Optional[AsyncLLMEngine] = None
metrics_collector = None
latency_tracker = None


class ChatMessage(BaseModel):
    """Chat message"""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = "default"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    n: int = 1


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    model: str = "default"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    n: int = 1


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]
    latency_ms: Optional[float] = None
    gpu_metrics: Optional[Dict] = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]
    latency_ms: Optional[float] = None
    gpu_metrics: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    gpu_available: bool
    engine_ready: bool
    metrics: Optional[Dict] = None


class MetricsResponse(BaseModel):
    """Metrics response"""
    system_metrics: Dict
    latency_stats: Dict
    engine_stats: Optional[Dict] = None


async def load_model(model_name: str = None, max_model_len: int = 4096):
    """Initialize vLLM engine"""
    global engine, metrics_collector, latency_tracker

    print("Initializing LLM inference service...")

    # Initialize metrics
    metrics_collector = MetricsCollector()
    latency_tracker = LatencyTracker()

    # Get model name from environment or use default
    if model_name is None:
        model_name = os.environ.get('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')

    print(f"Loading model: {model_name}")

    # Configure engine
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="auto",
    )

    # Create engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print("LLM inference engine ready!")
    return engine


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_name = os.environ.get('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')
    max_len = int(os.environ.get('MAX_MODEL_LEN', '4096'))
    await load_model(model_name, max_len)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_metrics = metrics_collector.get_gpu_metrics() if metrics_collector else {}

    return HealthResponse(
        status="healthy" if engine is not None else "initializing",
        model=os.environ.get('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct'),
        gpu_available=gpu_metrics.get('gpu_available', False),
        engine_ready=engine is not None,
        metrics=gpu_metrics
    )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    model_name = os.environ.get('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint
    Compatible with llmperf and OpenAI SDK
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start_time = time.time()

    # Convert messages to prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        n=request.n,
    )

    # Generate
    request_id = random_uuid()

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request_id, prompt, sampling_params),
            media_type="text/event-stream"
        )

    # Non-streaming
    results = []
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        results.append(request_output)

    final_output = results[-1]
    latency_ms = (time.time() - start_time) * 1000
    latency_tracker.record(latency_ms)

    # Build response
    choices = []
    total_tokens = 0
    for output in final_output.outputs:
        choices.append({
            "index": len(choices),
            "message": {
                "role": "assistant",
                "content": output.text
            },
            "finish_reason": output.finish_reason
        })
        total_tokens += len(output.token_ids)

    gpu_metrics = metrics_collector.get_gpu_metrics()

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage={
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": total_tokens,
            "total_tokens": len(final_output.prompt_token_ids) + total_tokens
        },
        latency_ms=latency_ms,
        gpu_metrics=gpu_metrics
    )


async def stream_chat_completion(
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams
) -> AsyncIterator[str]:
    """Stream chat completion responses"""
    start_time = time.time()
    first_token_time = None

    async for request_output in engine.generate(prompt, sampling_params, request_id):
        if first_token_time is None:
            first_token_time = time.time()

        for output in request_output.outputs:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "default",
                "choices": [{
                    "index": 0,
                    "delta": {"content": output.text},
                    "finish_reason": None
                }]
            }
            yield f"data: {chunk}\n\n"

    # Final chunk
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "default",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {final_chunk}\n\n"
    yield "data: [DONE]\n\n"

    # Record latency
    total_latency = (time.time() - start_time) * 1000
    latency_tracker.record(total_latency)


@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    """
    OpenAI-compatible completion endpoint
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start_time = time.time()

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        n=request.n,
    )

    # Generate
    request_id = random_uuid()

    if request.stream:
        return StreamingResponse(
            stream_completion(request_id, request.prompt, sampling_params),
            media_type="text/event-stream"
        )

    # Non-streaming
    results = []
    async for request_output in engine.generate(request.prompt, sampling_params, request_id):
        results.append(request_output)

    final_output = results[-1]
    latency_ms = (time.time() - start_time) * 1000
    latency_tracker.record(latency_ms)

    # Build response
    choices = []
    total_tokens = 0
    for output in final_output.outputs:
        choices.append({
            "index": len(choices),
            "text": output.text,
            "finish_reason": output.finish_reason
        })
        total_tokens += len(output.token_ids)

    gpu_metrics = metrics_collector.get_gpu_metrics()

    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage={
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": total_tokens,
            "total_tokens": len(final_output.prompt_token_ids) + total_tokens
        },
        latency_ms=latency_ms,
        gpu_metrics=gpu_metrics
    )


async def stream_completion(
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams
) -> AsyncIterator[str]:
    """Stream completion responses"""
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        for output in request_output.outputs:
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": "default",
                "choices": [{
                    "text": output.text,
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {chunk}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system and performance metrics"""
    system_metrics = metrics_collector.get_all_metrics()
    latency_stats = latency_tracker.get_stats()

    # Add engine stats if available
    engine_stats = {}
    # vLLM doesn't expose direct stats, but we can add custom ones

    return MetricsResponse(
        system_metrics=system_metrics,
        latency_stats=latency_stats,
        engine_stats=engine_stats
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
        "service": "LLM Inference API (vLLM)",
        "version": "1.0.0",
        "model": os.environ.get('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct'),
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions (POST)",
            "completions": "/v1/completions (POST)",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8000))

    print(f"Starting LLM inference service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
