"""
Benchmark Script for Latency Comparison
Compares performance between cluster and edge deployments
"""
import argparse
import time
import requests
import statistics
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from PIL import Image
import io
import numpy as np


class LatencyBenchmark:
    """Benchmark tool for measuring inference latency"""

    def __init__(self, cluster_url: str, edge_url: str = None):
        self.cluster_url = cluster_url.rstrip('/')
        self.edge_url = edge_url.rstrip('/') if edge_url else None
        self.test_images = []

    def generate_test_image(self, seed: int = None) -> bytes:
        """Generate a random test image"""
        if seed is not None:
            np.random.seed(seed)

        # Create random 32x32 RGB image
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes.getvalue()

    def measure_single_request(self, url: str, image_bytes: bytes) -> Dict:
        """Measure latency for a single inference request"""
        start_time = time.time()

        try:
            response = requests.post(
                f"{url}/predict",
                files={"file": ("test.png", image_bytes, "image/png")},
                timeout=30
            )
            response.raise_for_status()

            total_time = (time.time() - start_time) * 1000  # ms

            result = response.json()
            gpu_latency = result.get('latency_ms', 0)

            return {
                'success': True,
                'total_latency_ms': total_time,
                'gpu_inference_ms': gpu_latency,
                'network_latency_ms': total_time - gpu_latency,
                'prediction': result.get('top_class'),
                'confidence': result.get('confidence'),
                'status_code': response.status_code
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_latency_ms': (time.time() - start_time) * 1000
            }

    def run_sequential_benchmark(self, url: str, num_requests: int = 100) -> Dict:
        """Run sequential inference requests"""
        print(f"\nRunning sequential benchmark: {num_requests} requests")
        results = []

        for i in range(num_requests):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_requests}")

            image_bytes = self.generate_test_image(seed=i)
            result = self.measure_single_request(url, image_bytes)
            results.append(result)

            # Small delay to avoid overwhelming the server
            time.sleep(0.1)

        return self.analyze_results(results, "Sequential")

    def run_concurrent_benchmark(self, url: str, num_requests: int = 100,
                                 concurrency: int = 10) -> Dict:
        """Run concurrent inference requests"""
        print(f"\nRunning concurrent benchmark: {num_requests} requests, "
              f"concurrency={concurrency}")

        results = []
        images = [self.generate_test_image(seed=i) for i in range(num_requests)]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.measure_single_request, url, img)
                for img in images
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{num_requests}")

        return self.analyze_results(results, f"Concurrent (workers={concurrency})")

    def run_batch_benchmark(self, url: str, batch_sizes: List[int] = None) -> Dict:
        """Run batch inference benchmark"""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        print(f"\nRunning batch benchmark with sizes: {batch_sizes}")
        batch_results = {}

        for batch_size in batch_sizes:
            print(f"\n  Testing batch size: {batch_size}")

            # Generate images
            images = [self.generate_test_image(seed=i) for i in range(batch_size)]

            # Create files list
            files = [
                ("files", (f"test_{i}.png", img, "image/png"))
                for i, img in enumerate(images)
            ]

            start_time = time.time()

            try:
                response = requests.post(
                    f"{url}/predict/batch",
                    files=files,
                    timeout=60
                )
                response.raise_for_status()

                total_time = (time.time() - start_time) * 1000
                result = response.json()

                batch_results[batch_size] = {
                    'success': True,
                    'batch_size': batch_size,
                    'total_latency_ms': total_time,
                    'latency_per_image_ms': result.get('latency_per_image_ms'),
                    'throughput_images_per_sec': result.get('throughput_images_per_sec'),
                    'gpu_time_ms': result.get('total_latency_ms'),
                }

                print(f"    Throughput: {result.get('throughput_images_per_sec', 0):.2f} images/sec")

            except Exception as e:
                batch_results[batch_size] = {
                    'success': False,
                    'error': str(e)
                }

            time.sleep(1)  # Pause between batches

        return batch_results

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

        gpu_latencies = [r['gpu_inference_ms'] for r in successful if 'gpu_inference_ms' in r]
        network_latencies = [r['network_latency_ms'] for r in successful if 'network_latency_ms' in r]

        n = len(latencies)

        stats = {
            'benchmark': benchmark_name,
            'total_requests': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100,

            'total_latency': {
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

        if gpu_latencies:
            stats['gpu_latency'] = {
                'mean_ms': statistics.mean(gpu_latencies),
                'median_ms': statistics.median(gpu_latencies),
            }

        if network_latencies:
            stats['network_latency'] = {
                'mean_ms': statistics.mean(network_latencies),
                'median_ms': statistics.median(network_latencies),
            }

        return stats

    def compare_deployments(self, num_requests: int = 100):
        """Compare cluster vs edge deployment performance"""
        print("\n" + "="*70)
        print("LATENCY COMPARISON: Cluster vs Edge")
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
        if results['successful'] == 0:
            print("  No successful requests")
            return

        lat = results['total_latency']
        print(f"  Success Rate: {results['success_rate']:.1f}%")
        print(f"  Total Latency:")
        print(f"    Mean:   {lat['mean_ms']:8.2f} ms")
        print(f"    Median: {lat['median_ms']:8.2f} ms")
        print(f"    P95:    {lat['p95_ms']:8.2f} ms")
        print(f"    P99:    {lat['p99_ms']:8.2f} ms")
        print(f"    Min:    {lat['min_ms']:8.2f} ms")
        print(f"    Max:    {lat['max_ms']:8.2f} ms")
        print(f"    Stdev:  {lat['stdev_ms']:8.2f} ms")

        if 'gpu_latency' in results:
            print(f"  GPU Inference: {results['gpu_latency']['mean_ms']:.2f} ms")

        if 'network_latency' in results:
            print(f"  Network: {results['network_latency']['mean_ms']:.2f} ms")

    def print_comparison(self, cluster: Dict, edge: Dict):
        """Print comparison between cluster and edge"""
        c_lat = cluster['total_latency']['mean_ms']
        e_lat = edge['total_latency']['mean_ms']

        diff = e_lat - c_lat
        pct = (diff / c_lat) * 100

        print(f"  Cluster Mean Latency: {c_lat:.2f} ms")
        print(f"  Edge Mean Latency:    {e_lat:.2f} ms")
        print(f"  Difference:           {diff:+.2f} ms ({pct:+.1f}%)")

        if diff > 0:
            print(f"  → Edge is SLOWER by {diff:.2f} ms")
        else:
            print(f"  → Edge is FASTER by {abs(diff):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='ML Inference Latency Benchmark')
    parser.add_argument('--cluster-url', required=True, help='Cluster inference service URL')
    parser.add_argument('--edge-url', help='Edge inference service URL')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests')
    parser.add_argument('--concurrency', type=int, default=10, help='Concurrent workers')
    parser.add_argument('--mode', choices=['sequential', 'concurrent', 'batch', 'compare', 'all'],
                        default='compare', help='Benchmark mode')
    parser.add_argument('--output', help='Output file for results (JSON)')

    args = parser.parse_args()

    benchmark = LatencyBenchmark(args.cluster_url, args.edge_url)

    results = {}

    if args.mode in ['sequential', 'all']:
        results['sequential'] = benchmark.run_sequential_benchmark(
            args.cluster_url, args.requests
        )

    if args.mode in ['concurrent', 'all']:
        results['concurrent'] = benchmark.run_concurrent_benchmark(
            args.cluster_url, args.requests, args.concurrency
        )

    if args.mode in ['batch', 'all']:
        results['batch'] = benchmark.run_batch_benchmark(args.cluster_url)

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
