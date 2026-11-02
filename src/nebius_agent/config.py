"""Settings. Everything is an environment variable so a partner can drop this into any runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

TOKEN_FACTORY_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


@dataclass(frozen=True)
class Settings:
    base_url: str = field(default_factory=lambda: os.getenv("NEBIUS_BASE_URL", TOKEN_FACTORY_BASE_URL))
    api_key: str = field(default_factory=lambda: os.getenv("NEBIUS_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("NEBIUS_MODEL", DEFAULT_MODEL))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    daytona_api_key: str = field(default_factory=lambda: os.getenv("DAYTONA_API_KEY", ""))
    allow_local_exec: bool = field(default_factory=lambda: os.getenv("ALLOW_LOCAL_EXEC", "0") == "1")
    # tracing: console | file | otlp | none
    trace_exporter: str = field(default_factory=lambda: os.getenv("TRACE_EXPORTER", "file"))
    trace_file: str = field(default_factory=lambda: os.getenv("TRACE_FILE", "traces/spans.jsonl"))
    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "nebius-pydantic-ai-reference"))

    @property
    def is_mock(self) -> bool:
        return "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def validate(self) -> None:
        if not self.api_key and not self.is_mock:
            raise RuntimeError(
                "NEBIUS_API_KEY is not set. Create a key in the Token Factory console "
                "(https://tokenfactory.nebius.com) or point NEBIUS_BASE_URL at the mock server."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
