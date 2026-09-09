from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

import httpx

# Fixed prompt so results are comparable across models/pools/runs.
GENERATE_PROMPT = (
    "Write a Python function that takes a list of integers and returns the two "
    "numbers that sum to a given target, using a single-pass hash map approach. "
    "Include a docstring and a short usage example."
)

EMBED_INPUT = (
    "A distributed inference router directs reasoning requests across pooled "
    "GPU capacity, falling back to a smaller model when the primary pool is saturated."
)

# 1x1 black pixel PNG - exercises the vision code path without needing a real image.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@dataclass
class BenchmarkResult:
    timestamp: float
    pool_name: str
    port: int
    model: str
    kind: str  # "generate", "vision", "embed"
    success: bool
    error: Optional[str] = None
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    prompt_eval_count: int = 0
    prompt_eval_duration_ms: float = 0.0
    eval_count: int = 0
    eval_duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0
    response_chars: int = 0


class BenchmarkRunner:
    def __init__(self, timeout_seconds: float = 600.0):
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def run_generate(
        self, pool_name: str, port: int, model: str,
        prompt: str = GENERATE_PROMPT, images: Optional[list] = None
    ) -> BenchmarkResult:
        kind = "vision" if images else "generate"
        now = time.time()
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            if images:
                payload["images"] = images
            resp = await self.client.post(f"http://127.0.0.1:{port}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

            load_ms = data.get("load_duration", 0) / 1e6
            prompt_eval_ms = data.get("prompt_eval_duration", 0) / 1e6
            eval_count = data.get("eval_count", 0)
            eval_ms = data.get("eval_duration", 0) / 1e6

            return BenchmarkResult(
                timestamp=now, pool_name=pool_name, port=port, model=model, kind=kind,
                success=True,
                total_duration_ms=data.get("total_duration", 0) / 1e6,
                load_duration_ms=load_ms,
                prompt_eval_count=data.get("prompt_eval_count", 0),
                prompt_eval_duration_ms=prompt_eval_ms,
                eval_count=eval_count,
                eval_duration_ms=eval_ms,
                tokens_per_second=(eval_count / (eval_ms / 1000)) if eval_ms else 0.0,
                ttft_ms=load_ms + prompt_eval_ms,
                response_chars=len(data.get("response", "")),
            )
        except Exception as e:
            return BenchmarkResult(
                timestamp=now, pool_name=pool_name, port=port, model=model,
                kind=kind, success=False, error=str(e),
            )

    async def run_vision_smoke_test(self, pool_name: str, port: int, model: str) -> BenchmarkResult:
        return await self.run_generate(
            pool_name, port, model,
            prompt="Describe this image in one sentence.",
            images=[_TINY_PNG_B64],
        )

    async def run_embed(
        self, pool_name: str, port: int, model: str, text: str = EMBED_INPUT
    ) -> BenchmarkResult:
        now = time.time()
        try:
            start = time.perf_counter()
            resp = await self.client.post(
                f"http://127.0.0.1:{port}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            data = resp.json()
            embeddings = data.get("embeddings", [[]])
            dims = len(embeddings[0]) if embeddings else 0

            return BenchmarkResult(
                timestamp=now, pool_name=pool_name, port=port, model=model, kind="embed",
                success=True,
                total_duration_ms=data.get("total_duration", 0) / 1e6 or elapsed_ms,
                load_duration_ms=data.get("load_duration", 0) / 1e6,
                response_chars=dims,  # dimension count, reusing the field
                ttft_ms=elapsed_ms,
            )
        except Exception as e:
            return BenchmarkResult(
                timestamp=now, pool_name=pool_name, port=port, model=model,
                kind="embed", success=False, error=str(e),
            )

    async def close(self):
        await self.client.aclose()


def store_result(result: BenchmarkResult) -> int:
    from src.storage import insert
    return insert("benchmark_results", asdict(result))
