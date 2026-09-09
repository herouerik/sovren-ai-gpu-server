#!/usr/bin/env python3
"""Curated benchmark batch across all 3 ollama pools.

Sized to each pool's actual VRAM budget (see config.yaml gpu_ids):
  meta   - GPU0  (RTX 2080 Ti, 11GB)  - small models + vision/embed smoke tests
  pool-b - GPU5,6 (2x P100, 32GB)     - mid-size models
  pool-a - GPU1-4 (4x P100, 64GB)     - large reasoning models

Each pool runs its list sequentially (only one model fits loaded at a time per
pool) but the 3 pools run concurrently against each other since they use
disjoint GPUs. The pool's original production model is benchmarked first (as
a baseline) and reloaded last, so KEEP_ALIVE=-1 leaves the pool exactly as it
was before this script ran.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.benchmark import BenchmarkRunner, store_result

CURATED = {
    # deepseek-r1:14b intentionally excluded: GPU0's CUDA init is currently broken
    # (NVML/driver mismatch - see memory) and that model triggers a ~15-25min
    # GPU-discovery retry storm with no new information beyond what qwen2.5-coder
    # already showed (CPU fallback, degraded but functional). Re-add after reboot.
    "meta": ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen2.5-coder:14b-8k"],
    "pool-b": ["qwen3-coder:30b-sovereign", "qwen3-agent:latest", "devstral-small-2:latest",
               "gpt-oss:20b", "qwen3.6:latest"],
    "pool-a": ["qwen3-coder-next:sovereign-128k", "deepseek-r1-sovereign:70b",
               "deepseek-r1:70b", "llama4:scout"],
}


async def run_pool(runner: BenchmarkRunner, pool_name: str, port: int, original_model: str):
    candidates = CURATED.get(pool_name, [])
    # original model first (baseline) ... then candidates ... then original again (restore)
    order = [original_model] + [m for m in candidates if m != original_model] + [original_model]

    for model in order:
        t0 = time.monotonic()
        print(f"[{pool_name}] loading/running {model} ...", flush=True)
        result = await runner.run_generate(pool_name, port, model)
        store_result(result)
        elapsed = time.monotonic() - t0
        if result.success:
            print(f"[{pool_name}] {model}: {result.tokens_per_second:.1f} tok/s, "
                  f"ttft={result.ttft_ms:.0f}ms, total={elapsed:.1f}s", flush=True)
        else:
            print(f"[{pool_name}] {model}: FAILED - {result.error}", flush=True)

    if pool_name == "meta":
        print(f"[{pool_name}] vision smoke test: llama3.2-vision:11b ...", flush=True)
        vision_result = await runner.run_vision_smoke_test(pool_name, port, "llama3.2-vision:11b")
        store_result(vision_result)
        print(f"[{pool_name}] vision: {'ok' if vision_result.success else 'FAILED - ' + str(vision_result.error)}", flush=True)

        print(f"[{pool_name}] embed smoke test: nomic-embed-text ...", flush=True)
        embed_result = await runner.run_embed(pool_name, port, "nomic-embed-text")
        store_result(embed_result)
        print(f"[{pool_name}] embed: {'ok, dims=' + str(embed_result.response_chars) if embed_result.success else 'FAILED - ' + str(embed_result.error)}", flush=True)

        # restore meta to its idle state (no model was loaded before we started)
        print(f"[{pool_name}] restoring original model {original_model} ...", flush=True)
        restore = await runner.run_generate(pool_name, port, original_model)
        store_result(restore)


async def main():
    runner = BenchmarkRunner(timeout_seconds=900.0)
    services = {s.name: s for s in settings.get_ollama_services()}

    tasks = []
    for pool_name, svc in services.items():
        tasks.append(run_pool(runner, pool_name, svc.port, svc.model))

    start = time.monotonic()
    await asyncio.gather(*tasks)
    await runner.close()
    print(f"\nBenchmark batch complete in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
