from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class OllamaServiceConfig(BaseSettings):
    name: str
    port: int
    gpu_ids: List[int]
    model: str
    description: str = ""
    systemd_service: str = ""
    # The port LAN clients actually connect to. Usually equals `port`, but on
    # this box `port` is Ollama's real (internal-only) bind and a reverse
    # proxy sits in front on a different public port -- the collectors need
    # both: `port` to query /api/ps directly, `public_port` to watch for the
    # connection pileups a broken caller produces against the real front door.
    public_port: Optional[int] = None
    # Services that are intentionally not running right now (provisioned for
    # a future purpose, never started) shouldn't be reported as "unhealthy"
    # just because nothing answers on their port.
    enabled: bool = True

    model_config = SettingsConfigDict(extra="allow")

    def effective_public_port(self) -> int:
        return self.public_port if self.public_port is not None else self.port


class CollectorsConfig(BaseSettings):
    gpu_poll_interval_seconds: int = 2
    gpu_hardware_poll_interval_seconds: int = 60
    ollama_state_poll_interval_seconds: int = 5
    log_poll_interval_seconds: int = 1
    connections_poll_interval_seconds: int = 5

    model_config = SettingsConfigDict(extra="allow")


class StorageConfig(BaseSettings):
    db_path: str = "data/monitor.db"
    # Discrete events (patterns/alerts, load_cycles, benchmark_results):
    # retained for exactly this many calendar days via day-slots that wrap
    # and self-evict on write -- see storage.day_bucket_insert().
    event_retention_days: int = 14
    # Established-connection sample ring: window width and bucket
    # resolution -- see storage.connection_bucket_upsert().
    connection_window_hours: int = 48
    connection_bucket_seconds: int = 30
    # Fixed-capacity ring buffers for raw request/task-sample logs -- see
    # storage.ring_insert(). Sized generously above what any endpoint
    # actually queries (max window is 24h) so it always covers a full day
    # even at high traffic.
    raw_ring_capacity: int = 100_000
    # GPU load samples (util/mem/temp/power) are kept in memory only, never
    # persisted -- this many minutes of history per GPU.
    gpu_sample_memory_minutes: int = 240

    model_config = SettingsConfigDict(extra="allow")


class PatternsConfig(BaseSettings):
    latency_spike_threshold_ms: int = 5000
    swap_spike_multiplier: float = 5.0
    quota_exhaustion_keywords: List[str] = Field(default_factory=lambda: [
        "quota exceeded", "quota exhausted", "rate limit", "rate limited", "too many requests", "429"
    ])
    cpu_spillover_threshold_percent: int = 80
    baseline_window_minutes: int = 10
    reload_storm_count_threshold: int = 5
    reload_storm_window_seconds: int = 300
    ctx_churn_distinct_values_threshold: int = 2
    ctx_churn_window_seconds: int = 900
    near_zero_keep_alive_seconds: int = 5
    connection_pileup_threshold: int = 6

    model_config = SettingsConfigDict(extra="allow")


class PromptInsightConfig(BaseSettings):
    # Off by default: requires a reverse proxy mirroring requests to
    # /api/prompt_mirror (see README "Prompt insight") -- without that
    # nothing ever arrives here regardless of this flag.
    enabled: bool = False
    # Name of an entry in ollama.services to ask for a real ~8-word
    # summary (e.g. an otherwise-idle GPU dedicated to this). Looked up by
    # port only, independent of that service's own `enabled` flag -- not
    # reachable or not configured both just mean every prompt falls back
    # to mechanical truncation, no error.
    summarizer_service: Optional[str] = None
    summarizer_timeout_seconds: float = 20.0
    fallback_max_words: int = 40
    fallback_max_chars: int = 200
    ring_capacity: int = 100

    model_config = SettingsConfigDict(extra="allow")


class ServerConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    model_config = SettingsConfigDict(extra="allow")


def _load_yaml_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml_config = _load_yaml_config()


class Settings(BaseSettings):
    ollama: dict = Field(default_factory=dict)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    patterns: PatternsConfig = Field(default_factory=PatternsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    prompt_insight: PromptInsightConfig = Field(default_factory=PromptInsightConfig)

    model_config = SettingsConfigDict(extra="allow")

    def __init__(self, **kwargs):
        # Merge YAML config with any provided kwargs
        merged = {**_yaml_config, **kwargs}
        super().__init__(**merged)

    def get_ollama_services(self) -> List[OllamaServiceConfig]:
        services = self.ollama.get("services", [])
        return [OllamaServiceConfig(**s) for s in services]

    def get_ollama_service(self, name: str) -> Optional[OllamaServiceConfig]:
        return next((s for s in self.get_ollama_services() if s.name == name), None)


settings = Settings()