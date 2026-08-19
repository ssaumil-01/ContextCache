import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, Tuple

# Import psutil for resource monitoring
try:
    import psutil
except ImportError:
    print("psutil not installed. Installing it now...")
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil

# Configuration
TESTS = [
    {"name": "Test A (10 VUs)", "vus": 10, "script": "tools/load_testing/load_test_10.js"},
    {"name": "Test B (50 VUs)", "vus": 50, "script": "tools/load_testing/load_test_50.js"},
    {"name": "Test C (100 VUs)", "vus": 100, "script": "tools/load_testing/load_test_100.js"},
    {"name": "Test D (500 VUs)", "vus": 500, "script": "tools/load_testing/load_test_500.js"},
]

# Set test duration via environment or command-line (defaults to "2m" for full run)
TEST_DURATION = os.environ.get("TEST_DURATION", "2m")


class ResourceMonitor(threading.Thread):
    def __init__(self, server_pid: int):
        super().__init__()
        self.server_pid = server_pid
        self.stopped = threading.Event()
        self.cpu_samples = []
        self.mem_samples = []
        self.proc_cpu_samples = []
        self.proc_mem_samples = []

    def run(self):
        try:
            parent = psutil.Process(self.server_pid)
        except Exception:
            parent = None

        # Initialize CPU percentage trackers
        psutil.cpu_percent(interval=None)

        # Keep track of Process objects to preserve CPU baseline tracking
        monitored_processes = {}

        while not self.stopped.wait(1.0):
            # System Metrics
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.mem_samples.append(psutil.virtual_memory().percent)

            # Server Process & Children Metrics
            if parent:
                try:
                    # Get current active process list
                    children = parent.children(recursive=True)
                    current_pids = {p.pid for p in children}
                    current_pids.add(parent.pid)

                    # Remove terminated processes from cache
                    for pid in list(monitored_processes.keys()):
                        if pid not in current_pids:
                            monitored_processes.pop(pid)

                    # Add new processes to cache
                    if parent.pid not in monitored_processes:
                        # Initialize baseline
                        parent.cpu_percent(interval=None)
                        monitored_processes[parent.pid] = parent

                    for child in children:
                        if child.pid not in monitored_processes:
                            # Initialize baseline by calling cpu_percent once
                            try:
                                child.cpu_percent(interval=None)
                            except Exception:
                                pass
                            monitored_processes[child.pid] = child

                    # Sum metrics across all cached Process objects
                    proc_cpu = 0.0
                    proc_mem = 0.0
                    for pid, p in monitored_processes.items():
                        try:
                            proc_cpu += p.cpu_percent(interval=None)
                            proc_mem += p.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    self.proc_cpu_samples.append(proc_cpu)
                    self.proc_mem_samples.append(proc_mem / (1024 * 1024))  # RSS in MB
                except Exception:
                    pass

    def stop(self):
        self.stopped.set()

    def get_averages(self) -> Tuple[float, float, float, float]:
        sys_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        sys_mem = sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0.0
        proc_cpu = (
            sum(self.proc_cpu_samples) / len(self.proc_cpu_samples)
            if self.proc_cpu_samples
            else 0.0
        )
        proc_mem = (
            sum(self.proc_mem_samples) / len(self.proc_mem_samples)
            if self.proc_mem_samples
            else 0.0
        )
        return sys_cpu, sys_mem, proc_cpu, proc_mem


def flush_redis():
    """Flush Redis to ensure cache starts cold for each test."""
    print("Flushing Redis...")
    try:
        subprocess.run(["redis-cli", "flushall"], capture_output=True, check=True)
        print("Redis cache flushed successfully.")
    except Exception as e:
        print(f"Warning: Failed to flush Redis: {e}")


def start_server(workers: int, use_local_embeddings: bool = False) -> subprocess.Popen:
    """Start Uvicorn server in a subprocess."""
    print(
        f"Starting FastAPI server with workers={workers} (use_local_embeddings={use_local_embeddings})..."
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "tools.load_testing.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--workers",
        str(workers),
    ]
    env = os.environ.copy()
    env["USE_LOCAL_EMBEDDINGS"] = "True" if use_local_embeddings else "False"
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    # Wait for server startup
    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/health")
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("Server is healthy.")
                    return proc
        except Exception:
            pass
        time.sleep(0.5)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise RuntimeError(f"Server failed to start on port 8000 after {timeout} seconds.")


def scrape_metrics() -> Dict[str, Dict[str, float]]:
    """Scrape Prometheus metrics repeatedly to cover all active workers and capture X-PID."""
    metrics_data = {}
    # Scrape 50 times to ensure round-robin hits all worker PIDs
    for _ in range(50):
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/metrics")
            with urllib.request.urlopen(req, timeout=2) as resp:
                headers = dict(resp.info())
                # Get PID from headers
                pid = headers.get("X-PID") or headers.get("x-pid") or "default"
                body = resp.read().decode("utf-8")
                metrics_data[pid] = body
        except Exception:
            pass
        time.sleep(0.01)

    target_metrics = [
        "rag_cache_redis_latency_seconds_sum",
        "rag_cache_redis_latency_seconds_count",
        "rag_cache_faiss_search_latency_seconds_sum",
        "rag_cache_faiss_search_latency_seconds_count",
        "rag_cache_embedding_latency_seconds_sum",
        "rag_cache_embedding_latency_seconds_count",
    ]

    parsed = {}
    for pid, body in metrics_data.items():
        pid_metrics = {m: 0.0 for m in target_metrics}
        for line in body.splitlines():
            if line.startswith("#"):
                continue
            for metric in target_metrics:
                if line.startswith(metric):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            val = float(parts[-1])
                            pid_metrics[metric] += val
                        except ValueError:
                            pass
        parsed[pid] = pid_metrics
    return parsed


def compute_metrics_delta(
    baseline: Dict[str, Dict[str, float]], ending: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compute baseline vs ending latency differences and return averages in ms."""
    delta = {
        "redis_sum": 0.0,
        "redis_count": 0.0,
        "faiss_sum": 0.0,
        "faiss_count": 0.0,
        "embedding_sum": 0.0,
        "embedding_count": 0.0,
    }

    all_pids = set(baseline.keys()).union(set(ending.keys()))
    for pid in all_pids:
        b_pid = baseline.get(pid, {})
        e_pid = ending.get(pid, {})

        delta["redis_sum"] += max(
            0.0,
            e_pid.get("rag_cache_redis_latency_seconds_sum", 0.0)
            - b_pid.get("rag_cache_redis_latency_seconds_sum", 0.0),
        )
        delta["redis_count"] += max(
            0.0,
            e_pid.get("rag_cache_redis_latency_seconds_count", 0.0)
            - b_pid.get("rag_cache_redis_latency_seconds_count", 0.0),
        )

        delta["faiss_sum"] += max(
            0.0,
            e_pid.get("rag_cache_faiss_search_latency_seconds_sum", 0.0)
            - b_pid.get("rag_cache_faiss_search_latency_seconds_sum", 0.0),
        )
        delta["faiss_count"] += max(
            0.0,
            e_pid.get("rag_cache_faiss_search_latency_seconds_count", 0.0)
            - b_pid.get("rag_cache_faiss_search_latency_seconds_count", 0.0),
        )

        delta["embedding_sum"] += max(
            0.0,
            e_pid.get("rag_cache_embedding_latency_seconds_sum", 0.0)
            - b_pid.get("rag_cache_embedding_latency_seconds_sum", 0.0),
        )
        delta["embedding_count"] += max(
            0.0,
            e_pid.get("rag_cache_embedding_latency_seconds_count", 0.0)
            - b_pid.get("rag_cache_embedding_latency_seconds_count", 0.0),
        )

    redis_avg = (
        (delta["redis_sum"] / delta["redis_count"]) * 1000 if delta["redis_count"] > 0 else 0.0
    )
    faiss_avg = (
        (delta["faiss_sum"] / delta["faiss_count"]) * 1000 if delta["faiss_count"] > 0 else 0.0
    )
    emb_avg = (
        (delta["embedding_sum"] / delta["embedding_count"]) * 1000
        if delta["embedding_count"] > 0
        else 0.0
    )

    return {
        "redis_latency_ms": redis_avg,
        "faiss_latency_ms": faiss_avg,
        "embedding_latency_ms": emb_avg,
    }


def run_single_benchmark(
    test_config: Dict[str, Any], workers: int, use_local_embeddings: bool = False
) -> Dict[str, Any]:
    """Run a single benchmark configuration and return results."""
    print("\n" + "=" * 50)
    print(
        f"RUNNING: {test_config['name']} with workers={workers} (use_local_embeddings={use_local_embeddings}, Duration: {TEST_DURATION})"
    )
    print("=" * 50)

    flush_redis()
    server_proc = start_server(workers, use_local_embeddings)

    # Baseline Metrics
    baseline_metrics = scrape_metrics()

    # Start Resource Monitor
    monitor = ResourceMonitor(server_proc.pid)
    monitor.start()

    # Run k6
    summary_path = "tools/load_testing/temp_summary.json"
    if os.path.exists(summary_path):
        os.remove(summary_path)

    k6_cmd = [
        "k6",
        "run",
        "--duration",
        TEST_DURATION,
        f"--summary-export={summary_path}",
        test_config["script"],
    ]

    print(f"Executing: {' '.join(k6_cmd)}")
    k6_proc = subprocess.run(k6_cmd, capture_output=True, text=True)
    if k6_proc.returncode != 0:
        print(f"k6 exited with code {k6_proc.returncode}")
    print(f"k6 stdout:\n{k6_proc.stdout}\n")
    print(f"k6 stderr:\n{k6_proc.stderr}\n")

    # Stop Monitor
    monitor.stop()
    monitor.join()

    # Ending Metrics
    ending_metrics = scrape_metrics()

    # Shutdown server
    print("Stopping FastAPI server...")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    # Read k6 output
    k6_data = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                k6_data = json.load(f)
            os.remove(summary_path)
        except Exception as e:
            print(f"Failed to read summary JSON: {e}")

    # Parse metrics
    metrics = k6_data.get("metrics", {})
    throughput = metrics.get("http_reqs", {}).get("rate", 0.0)

    # Latencies in milliseconds
    duration_metrics = metrics.get("http_req_duration", {})
    avg_latency = duration_metrics.get("avg", 0.0)
    p50_latency = duration_metrics.get("med", 0.0)
    p90_latency = duration_metrics.get("p(90)", 0.0)
    p95_latency = duration_metrics.get("p(95)", 0.0)

    # Errors
    error_rate = metrics.get("http_req_failed", {}).get("value", 0.0) * 100

    # Resource metrics
    sys_cpu, sys_mem, proc_cpu, proc_mem = monitor.get_averages()

    # Prometheus deltas
    deltas = compute_metrics_delta(baseline_metrics, ending_metrics)

    return {
        "vus": test_config["vus"],
        "throughput": throughput,
        "avg_ms": avg_latency,
        "p50_ms": p50_latency,
        "p90_ms": p90_latency,
        "p95_ms": p95_latency,
        "error_rate": error_rate,
        "sys_cpu": sys_cpu,
        "sys_mem": sys_mem,
        "proc_cpu": proc_cpu,
        "proc_mem": proc_mem,
        "redis_ms": deltas["redis_latency_ms"],
        "faiss_ms": deltas["faiss_latency_ms"],
        "embedding_ms": deltas["embedding_latency_ms"],
    }


def main():
    print(f"Starting benchmark runner. Scenario duration: {TEST_DURATION}")

    # 1. Run single-worker benchmarks
    print("\nRunning single-worker benchmarks...")
    single_results = []
    for test in TESTS:
        res = run_single_benchmark(test, workers=1, use_local_embeddings=False)
        single_results.append(res)

    # 2. Run multi-worker benchmarks (mock embeddings)
    print("\nRunning multi-worker (workers=4, mock embeddings) benchmarks...")
    multi_results = []
    for test in TESTS:
        res = run_single_benchmark(test, workers=4, use_local_embeddings=False)
        multi_results.append(res)

    # 3. Run multi-worker benchmarks (real embeddings)
    print("\nRunning multi-worker (workers=4, real SentenceTransformer) benchmarks...")
    real_results = []
    for test in TESTS:
        res = run_single_benchmark(test, workers=4, use_local_embeddings=True)
        real_results.append(res)

    # 4. Generate Consolidated Report
    report_path = "tools/load_testing/benchmark_results.md"
    print(f"\nGenerating consolidated report at {report_path}...")

    # Format functions
    def f_sec(ms):
        return f"{ms/1000:.3f}s"

    def f_ms(ms):
        return f"{ms:.2f}ms"

    def f_pct(val):
        return f"{val:.1f}%"

    report = []
    report.append("# ContextCache Concurrency Stress Test Results & Bottleneck Analysis")
    report.append(
        f"\n*Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} (Scenario Duration: {TEST_DURATION})*"
    )

    # Table 1: Single Worker Server
    report.append("\n## 1. Single Worker Benchmark Results (`--workers 1`, Mock Embeddings)")
    report.append(
        "\n| VUs | Throughput | Avg Latency | P50 (Med) | P90 | P95 | Errors | Sys CPU | Proc CPU | RSS Mem | Redis Latency | FAISS Latency | Embedding Latency |"
    )
    report.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    for r in single_results:
        report.append(
            f"| {r['vus']} | {r['throughput']:.2f} req/s | {f_sec(r['avg_ms'])} | {f_sec(r['p50_ms'])} | {f_sec(r['p90_ms'])} | {f_sec(r['p95_ms'])} | {f_pct(r['error_rate'])} | "
            f"{f_pct(r['sys_cpu'])} | {f_pct(r['proc_cpu'])} | {r['proc_mem']:.1f} MB | {f_ms(r['redis_ms'])} | {f_ms(r['faiss_ms'])} | {f_ms(r['embedding_ms'])} |"
        )

    # Table 2: Multi-worker Server (Mock)
    report.append("\n## 2. Multi-Worker Benchmark Results (`--workers 4`, Mock Embeddings)")
    report.append(
        "\n| VUs | Throughput | Avg Latency | P50 (Med) | P90 | P95 | Errors | Sys CPU | Proc CPU | RSS Mem | Redis Latency | FAISS Latency | Embedding Latency |"
    )
    report.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    for r in multi_results:
        report.append(
            f"| {r['vus']} | {r['throughput']:.2f} req/s | {f_sec(r['avg_ms'])} | {f_sec(r['p50_ms'])} | {f_sec(r['p90_ms'])} | {f_sec(r['p95_ms'])} | {f_pct(r['error_rate'])} | "
            f"{f_pct(r['sys_cpu'])} | {f_pct(r['proc_cpu'])} | {r['proc_mem']:.1f} MB | {f_ms(r['redis_ms'])} | {f_ms(r['faiss_ms'])} | {f_ms(r['embedding_ms'])} |"
        )

    # Table 3: Multi-worker Server (Real)
    report.append(
        "\n## 3. Multi-Worker Benchmark Results (`--workers 4`, Real SentenceTransformer)"
    )
    report.append(
        "\n| VUs | Throughput | Avg Latency | P50 (Med) | P90 | P95 | Errors | Sys CPU | Proc CPU | RSS Mem | Redis Latency | FAISS Latency | Embedding Latency |"
    )
    report.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    for r in real_results:
        report.append(
            f"| {r['vus']} | {r['throughput']:.2f} req/s | {f_sec(r['avg_ms'])} | {f_sec(r['p50_ms'])} | {f_sec(r['p90_ms'])} | {f_sec(r['p95_ms'])} | {f_pct(r['error_rate'])} | "
            f"{f_pct(r['sys_cpu'])} | {f_pct(r['proc_cpu'])} | {r['proc_mem']:.1f} MB | {f_ms(r['redis_ms'])} | {f_ms(r['faiss_ms'])} | {f_ms(r['embedding_ms'])} |"
        )

    # Table 4: Contention Comparison
    report.append("\n## 4. Worker Contention Comparison (Single vs. Multi-Worker, Mock Embeddings)")
    report.append(
        "\n| VUs | 1 Worker Throughput | 4 Workers Throughput | Speedup | 1 Worker P95 | 4 Workers P95 | P95 Latency Reduction | 1 Worker CPU | 4 Workers CPU |"
    )
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s, m in zip(single_results, multi_results):
        speedup = m["throughput"] / s["throughput"] if s["throughput"] > 0 else 0.0
        reduction = ((s["p95_ms"] - m["p95_ms"]) / s["p95_ms"]) * 100 if s["p95_ms"] > 0 else 0.0
        report.append(
            f"| {s['vus']} | {s['throughput']:.2f} req/s | {m['throughput']:.2f} req/s | {speedup:.2f}x | {f_sec(s['p95_ms'])} | {f_sec(m['p95_ms'])} | {f_pct(reduction)} | {f_pct(s['proc_cpu'])} | {f_pct(m['proc_cpu'])} |"
        )

    # Table 5: Embedding Overhead Comparison
    report.append(
        "\n## 5. Embedding Overhead Comparison (Mock vs. Real Embeddings under 4 Workers)"
    )
    report.append(
        "\n| VUs | Mock Throughput | Real Throughput | Throughput Retention | Mock P95 | Real P95 | Latency Overhead | Mock CPU | Real CPU |"
    )
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m, r in zip(multi_results, real_results):
        retention = (r["throughput"] / m["throughput"]) * 100 if m["throughput"] > 0 else 0.0
        overhead = ((r["p95_ms"] - m["p95_ms"]) / m["p95_ms"]) * 100 if m["p95_ms"] > 0 else 0.0
        report.append(
            f"| {m['vus']} | {m['throughput']:.2f} req/s | {r['throughput']:.2f} req/s | {f_pct(retention)} | {f_sec(m['p95_ms'])} | {f_sec(r['p95_ms'])} | {f_pct(overhead)} | {f_pct(m['proc_cpu'])} | {f_pct(r['proc_cpu'])} |"
        )

    # Bottleneck Analysis section
    report.append("\n## 6. Bottleneck & Latency Analysis")

    final_single = single_results[-1]
    final_multi = multi_results[-1]
    final_real = real_results[-1]

    report.append("\n### Quantitative Diagnoses:")

    # 1. CPU Saturation
    report.append(f"- **Uvicorn Worker CPU Utilisation (Test D, 500 VUs)**:")
    report.append(f"  - 1 Worker (Mock): **{final_single['proc_cpu']:.1f}%**")
    report.append(f"  - 4 Workers (Mock): **{final_multi['proc_cpu']:.1f}%**")
    report.append(f"  - 4 Workers (Real SentenceTransformer): **{final_real['proc_cpu']:.1f}%**")

    # 2. Redis, FAISS, Embedding latencies
    report.append(f"- **Sub-Component Latency Comparison (Test D, 500 VUs)**:")
    report.append(
        f"  - Redis Latency: Mock = **{final_multi['redis_ms']:.2f}ms**, Real = **{final_real['redis_ms']:.2f}ms**"
    )
    report.append(
        f"  - FAISS Latency: Mock = **{final_multi['faiss_ms']:.2f}ms**, Real = **{final_real['faiss_ms']:.2f}ms**"
    )
    report.append(
        f"  - Embedding Generation Latency: Mock = **{final_multi['embedding_ms']:.2f}ms**, Real = **{final_real['embedding_ms']:.2f}ms**"
    )

    # 3. Queueing delay vs Service time
    mock_service_time = (
        final_multi["redis_ms"] + final_multi["faiss_ms"] + final_multi["embedding_ms"]
    )
    real_service_time = final_real["redis_ms"] + final_real["faiss_ms"] + final_real["embedding_ms"]
    report.append(f"- **Total API Service Time vs. Client-Perceived Latency (Test D, 500 VUs)**:")
    report.append(
        f"  - Mock Embeddings: Service Time = **{mock_service_time:.2f}ms**, Client Latency (Avg) = **{f_sec(final_multi['avg_ms'])}**"
    )
    report.append(
        f"  - Real Embeddings: Service Time = **{real_service_time:.2f}ms**, Client Latency (Avg) = **{f_sec(final_real['avg_ms'])}**"
    )

    report.append("\n### Analysis & Conclusion:")

    if final_real["embedding_ms"] > 5.0:
        report.append(
            "1. **Is Embedding Generation a Bottleneck?** **YES, absolutely.**\n"
            f"   - **Evidence**: Real embedding generation latency rose to **{final_real['embedding_ms']:.2f}ms** (compared to **{final_multi['embedding_ms']:.2f}ms** for Mock). This represents a major increase in the actual core request service time. Under heavy CPU load, local SentenceTransformer embedding computation pins the multi-core CPU capacity, increasing client response times and degrading throughput.\n"
            f"2. **CPU Saturation & Worker Contention**: Real embeddings make the Uvicorn workers CPU-saturated much faster, aggravating connection backlog queueing. Worker CPU utilization hit **{final_real['proc_cpu']:.1f}%**, meaning the CPU is the hard limit when doing on-server neural inference.\n"
            "3. **Architectural Recommendation**: For high-concurrency production deployments, offload embedding generation to a dedicated GPU microservice or a scalable external API. This isolates the CPU-heavy neural network inference from the Uvicorn HTTP server loop, allowing ContextCache to maintain its sub-millisecond retrieval and routing throughput."
        )
    else:
        report.append(
            "1. **Is Embedding Generation a Bottleneck?** **NO, it remains lightweight under all-MiniLM-L6-v2.**\n"
            f"   - **Evidence**: Embedding latency remained at **{final_real['embedding_ms']:.2f}ms**, showing that the CPU cost of the neural model is minor compared to connection management and loop overheads.\n"
            "2. **Uvicorn Worker Contention**: The GIL and single-process architecture of Uvicorn remain the dominant bottleneck."
        )

    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"Benchmark run complete. Report saved to {report_path}.")


if __name__ == "__main__":
    main()
