"""
LLM Benchmark Script for Latency Comparison
Compares performance between cluster and edge deployments
Uses llmperf-style metrics: TTFT, throughput, latency
"""
import argparse
import time
import requests
import statistics
import json
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from openai import OpenAI


class LLMBenchmark:
    """Benchmark tool for measuring LLM inference performance"""

    def __init__(self, cluster_url: str, edge_url: Optional[str] = None):
        self.cluster_url = cluster_url.rstrip('/')
        self.edge_url = edge_url.rstrip('/') if edge_url else None

        # Test prompts of varying lengths
        self.test_prompts = self.generate_test_prompts()

    def generate_test_prompts(self) -> List[str]:
        """Generate test prompts of different lengths"""
        prompts = [
            # Short prompt (~10 tokens)
            "What is 2+2?",

            # Medium prompt (~50 tokens)
            "Explain the concept of machine learning in simple terms. "
            "What are the main types of machine learning algorithms?",

            # Long prompt (~100 tokens)
            "Write a detailed explanation of how neural networks work, "
            "including the concepts of forward propagation, backpropagation, "
            "activation functions, and gradient descent. Make sure to explain "
            "how these components work together to enable the network to learn.",

            # Very long prompt (~200 tokens)
            "Describe the architecture and training process of large language models "
            "like GPT. Explain the transformer architecture, attention mechanisms, "
            "positional encodings, and the pre-training and fine-tuning processes. "
            "Discuss the challenges of scaling these models to billions of parameters "
            "and the techniques used to make inference efficient, such as quantization, "
            "knowledge distillation, and speculative decoding. Also mention the "
            "environmental and computational costs associated with training these models."
        ]
        return prompts

    def measure_single_request(
        self,
        url: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> Dict:
        """Measure latency for a single LLM inference request"""
        start_time = time.time()

        try:
            # OpenAI-compatible API call
            response = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()

            total_time_ms = (time.time() - start_time) * 1000
            result = response.json()

            # Extract metrics
            usage = result.get('usage', {})
            latency_ms = result.get('latency_ms', total_time_ms)

            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

            # Calculate throughput
            tokens_per_second = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0

            return {
                'success': True,
                'total_latency_ms': total_time_ms,
                'inference_latency_ms': latency_ms,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'tokens_per_second': tokens_per_second,
                'response': result.get('choices', [{}])[0].get('message', {}).get('content', ''),
                'gpu_metrics': result.get('gpu_metrics', {})
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_latency_ms': (time.time() - start_time) * 1000
            }

    def measure_streaming_request(
        self,
        url: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> Dict:
        """Measure latency for streaming LLM inference (TTFT)"""
        start_time = time.time()
        first_token_time = None
        token_times = []

        try:
            response = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True
                },
                timeout=120,
                stream=True
            )
            response.raise_for_status()

            tokens = []
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break

                        current_time = time.time()

                        if first_token_time is None:
                            first_token_time = current_time
                            ttft_ms = (first_token_time - start_time) * 1000

                        token_times.append(current_time)

            total_time_ms = (time.time() - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms

            # Calculate inter-token latency
            if len(token_times) > 1:
                inter_token_latencies = [
                    (token_times[i] - token_times[i-1]) * 1000
                    for i in range(1, len(token_times))
                ]
                avg_inter_token_latency = statistics.mean(inter_token_latencies)
            else:
                avg_inter_token_latency = 0

            num_tokens = len(token_times)
            tokens_per_second = num_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

            return {
                'success': True,
                'total_latency_ms': total_time_ms,
                'ttft_ms': ttft_ms,  # Time to First Token
                'avg_inter_token_latency_ms': avg_inter_token_latency,
                'num_tokens': num_tokens,
                'tokens_per_second': tokens_per_second
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_latency_ms': (time.time() - start_time) * 1000
            }

    def run_sequential_benchmark(
        self,
        url: str,
        num_requests: int = 50,
        max_tokens: int = 256
    ) -> Dict:
        """Run sequential LLM inference requests"""
        print(f"\nRunning sequential benchmark: {num_requests} requests")
        results = []

        for i in range(num_requests):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_requests}")

            # Rotate through test prompts
            prompt = self.test_prompts[i % len(self.test_prompts)]
            result = self.measure_single_request(url, prompt, max_tokens)
            results.append(result)

            # Small delay
            time.sleep(0.5)

        return self.analyze_results(results, "Sequential")

    def run_concurrent_benchmark(
        self,
        url: str,
        num_requests: int = 50,
        concurrency: int = 5,
        max_tokens: int = 256
    ) -> Dict:
        """Run concurrent LLM inference requests"""
        print(f"\nRunning concurrent benchmark: {num_requests} requests, "
              f"concurrency={concurrency}")

        results = []
        prompts = [self.test_prompts[i % len(self.test_prompts)] for i in range(num_requests)]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.measure_single_request, url, prompt, max_tokens)
                for prompt in prompts
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{num_requests}")

        return self.analyze_results(results, f"Concurrent (workers={concurrency})")

    def run_streaming_benchmark(
        self,
        url: str,
        num_requests: int = 20,
        max_tokens: int = 256
    ) -> Dict:
        """Run streaming inference benchmark (measures TTFT)"""
        print(f"\nRunning streaming benchmark: {num_requests} requests")
        results = []

        for i in range(num_requests):
            if i % 5 == 0:
                print(f"  Progress: {i}/{num_requests}")

            prompt = self.test_prompts[i % len(self.test_prompts)]
            result = self.measure_streaming_request(url, prompt, max_tokens)
            results.append(result)

            time.sleep(0.5)

        return self.analyze_streaming_results(results, "Streaming")

    def analyze_results(self, results: List[Dict], benchmark_name: str) -> Dict:
        """Analyze benchmark results"""
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]

        if not successful:
            return {
                'benchmark': benchmark_name,
                'total_requests': len(results),
                'successful': 0,
                'failed': len(failed),
                'success_rate': 0.0
            }

        latencies = [r['total_latency_ms'] for r in successful]
        latencies.sort()

        tokens_per_sec = [r['tokens_per_second'] for r in successful if 'tokens_per_second' in r]
        completion_tokens = [r['completion_tokens'] for r in successful if 'completion_tokens' in r]

        n = len(latencies)

        stats = {
            'benchmark': benchmark_name,
            'total_requests': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100,

            'latency': {
                'mean_ms': statistics.mean(latencies),
                'median_ms': statistics.median(latencies),
                'stdev_ms': statistics.stdev(latencies) if n > 1 else 0,
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'p50_ms': latencies[n // 2],
                'p95_ms': latencies[int(n * 0.95)],
                'p99_ms': latencies[int(n * 0.99)],
            }
        }

        if tokens_per_sec:
            stats['throughput'] = {
                'mean_tokens_per_sec': statistics.mean(tokens_per_sec),
                'median_tokens_per_sec': statistics.median(tokens_per_sec),
                'total_tokens': sum(completion_tokens) if completion_tokens else 0,
                'avg_completion_tokens': statistics.mean(completion_tokens) if completion_tokens else 0
            }

        return stats

    def analyze_streaming_results(self, results: List[Dict], benchmark_name: str) -> Dict:
        """Analyze streaming benchmark results"""
        successful = [r for r in results if r.get('success')]

        if not successful:
            return {
                'benchmark': benchmark_name,
                'total_requests': len(results),
                'successful': 0
            }

        ttfts = [r['ttft_ms'] for r in successful]
        inter_token_latencies = [r['avg_inter_token_latency_ms'] for r in successful]
        tokens_per_sec = [r['tokens_per_second'] for r in successful]

        return {
            'benchmark': benchmark_name,
            'successful': len(successful),

            'ttft': {
                'mean_ms': statistics.mean(ttfts),
                'median_ms': statistics.median(ttfts),
                'p95_ms': sorted(ttfts)[int(len(ttfts) * 0.95)],
            },

            'inter_token_latency': {
                'mean_ms': statistics.mean(inter_token_latencies),
                'median_ms': statistics.median(inter_token_latencies),
            },

            'throughput': {
                'mean_tokens_per_sec': statistics.mean(tokens_per_sec),
                'median_tokens_per_sec': statistics.median(tokens_per_sec),
            }
        }

    def compare_deployments(self, num_requests: int = 50):
        """Compare cluster vs edge deployment performance"""
        print("\n" + "="*70)
        print("LLM LATENCY COMPARISON: Cluster vs Edge")
        print("="*70)

        # Test cluster
        print("\n--- Testing CLUSTER deployment ---")
        cluster_results = self.run_sequential_benchmark(self.cluster_url, num_requests)

        # Test edge (if available)
        edge_results = None
        if self.edge_url:
            print("\n--- Testing EDGE deployment ---")
            edge_results = self.run_sequential_benchmark(self.edge_url, num_requests)

        # Print comparison
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)

        print("\n--- CLUSTER ---")
        self.print_stats(cluster_results)

        if edge_results:
            print("\n--- EDGE ---")
            self.print_stats(edge_results)

            print("\n--- COMPARISON ---")
            self.print_comparison(cluster_results, edge_results)

        return {
            'cluster': cluster_results,
            'edge': edge_results
        }

    def print_stats(self, results: Dict):
        """Print formatted statistics"""
        if results.get('successful', 0) == 0:
            print("  No successful requests")
            return

        lat = results['latency']
        print(f"  Success Rate: {results['success_rate']:.1f}%")
        print(f"  Latency:")
        print(f"    Mean:   {lat['mean_ms']:8.2f} ms")
        print(f"    Median: {lat['median_ms']:8.2f} ms")
        print(f"    P95:    {lat['p95_ms']:8.2f} ms")
        print(f"    P99:    {lat['p99_ms']:8.2f} ms")

        if 'throughput' in results:
            thr = results['throughput']
            print(f"  Throughput:")
            print(f"    Mean:   {thr['mean_tokens_per_sec']:8.2f} tokens/sec")
            print(f"    Median: {thr['median_tokens_per_sec']:8.2f} tokens/sec")
            print(f"    Total tokens: {thr['total_tokens']}")

    def print_comparison(self, cluster: Dict, edge: Dict):
        """Print comparison between cluster and edge"""
        c_lat = cluster['latency']['mean_ms']
        e_lat = edge['latency']['mean_ms']

        diff = e_lat - c_lat
        pct = (diff / c_lat) * 100

        print(f"  Cluster Mean Latency: {c_lat:.2f} ms")
        print(f"  Edge Mean Latency:    {e_lat:.2f} ms")
        print(f"  Difference:           {diff:+.2f} ms ({pct:+.1f}%)")

        if 'throughput' in cluster and 'throughput' in edge:
            c_thr = cluster['throughput']['mean_tokens_per_sec']
            e_thr = edge['throughput']['mean_tokens_per_sec']
            print(f"\n  Cluster Throughput: {c_thr:.2f} tokens/sec")
            print(f"  Edge Throughput:    {e_thr:.2f} tokens/sec")


def main():
    parser = argparse.ArgumentParser(description='LLM Inference Latency Benchmark')
    parser.add_argument('--cluster-url', required=True, help='Cluster inference service URL')
    parser.add_argument('--edge-url', help='Edge inference service URL')
    parser.add_argument('--requests', type=int, default=50, help='Number of requests')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent workers')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--mode', choices=['sequential', 'concurrent', 'streaming', 'compare', 'all'],
                        default='compare', help='Benchmark mode')
    parser.add_argument('--output', help='Output file for results (JSON)')

    args = parser.parse_args()

    benchmark = LLMBenchmark(args.cluster_url, args.edge_url)

    results = {}

    if args.mode in ['sequential', 'all']:
        results['sequential'] = benchmark.run_sequential_benchmark(
            args.cluster_url, args.requests, args.max_tokens
        )

    if args.mode in ['concurrent', 'all']:
        results['concurrent'] = benchmark.run_concurrent_benchmark(
            args.cluster_url, args.requests, args.concurrency, args.max_tokens
        )

    if args.mode in ['streaming', 'all']:
        results['streaming'] = benchmark.run_streaming_benchmark(
            args.cluster_url, args.requests // 2, args.max_tokens
        )

    if args.mode in ['compare', 'all']:
        if not args.edge_url:
            print("Error: --edge-url required for comparison mode")
            return
        results['comparison'] = benchmark.compare_deployments(args.requests)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
