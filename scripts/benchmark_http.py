"""Compare both engines through the same localhost HTTP server and client."""

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import socket
import threading
import time

import httpx
import torch
import uvicorn

from laya_blackwell.benchmark import BACKENDS, load_backend, summarize
from laya_blackwell.engine import hardware_info, model_path, REVISION
from laya_blackwell.server import create_app
from laya_blackwell.workloads import workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["upstream", "fused"])
    parser.add_argument("--output", default="results/benchmark-http.json")
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("Use at least 2 iterations")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(), "revision": REVISION,
        "packages": {p: version(p) for p in ("torch", "transformers", "laya", "httpx", "uvicorn")},
        "iterations": args.iterations, "warmup": 5,
        "method": "Both backends use the identical FastAPI app, run_in_threadpool, uvicorn, HTTP/1.1 localhost keep-alive client and JSON request. TCP_NODELAY enabled on the listener for all backends. Serial requests; client-observed wall time includes response decoding. No answer cache. Model load, server startup and 5 warmups excluded equally. Backends run sequentially. This tests a shared wrapper, not upstream's separately shipped server.",
        "rows": [],
    }
    if "upstream-fast" in args.backends:
        report["packages"]["tilelang"] = version("tilelang")
    path = model_path()
    for backend in args.backends:
        engine = load_backend(backend, path)
        app = create_app(engine=engine, backend=backend, api_key="", warmup=False)
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP) as listener:
            listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            worker.start()
            try:
                deadline = time.monotonic() + 30
                while not server.started:
                    if not worker.is_alive() or time.monotonic() > deadline:
                        raise RuntimeError("Benchmark server failed to start")
                    time.sleep(0.01)
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False) as client:
                    for batch, length in ((1, "short"), (16, "short"), (16, "long")):
                        request = workload(batch, length)
                        for _ in range(5):
                            client.post("/v1/systemone", json=request).raise_for_status()
                        samples = []
                        for _ in range(args.iterations):
                            start = time.perf_counter()
                            response = client.post("/v1/systemone", json=request)
                            response.raise_for_status()
                            payload = response.json()
                            samples.append((time.perf_counter() - start) * 1000)
                        if engine.device.type != "cuda":
                            raise RuntimeError("CPU fallback invalidates the benchmark")
                        row = {"backend": backend, "questions": batch, "state_length": length,
                               "input_tokens": payload["usage"]["input_tokens"],
                               **summarize(samples), "samples_ms": samples}
                        report["rows"].append(row)
                        print(f"{backend}: q={batch} {length} HTTP p50={row['p50_ms']:.3f} ms", flush=True)
            finally:
                server.should_exit = True
                worker.join(timeout=30)
                if worker.is_alive():
                    raise RuntimeError("Benchmark server did not stop")
                if hasattr(engine, "close"):
                    engine.close()
        del engine, app, server, worker
        gc.collect()
        torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
